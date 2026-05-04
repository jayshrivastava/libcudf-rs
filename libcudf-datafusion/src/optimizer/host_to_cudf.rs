use crate::aggregate::try_as_cudf_aggregate;
use crate::optimizer::CuDFConfig;
use crate::physical::{
    is_cudf_plan, try_as_cudf_hash_join, CuDFCoalesceBatchesExec, CuDFFilterExec, CuDFLoadExec,
    CuDFProjectionExec, CuDFSortExec, CuDFUnloadExec,
};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::DataFusionError;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::ExecutionPlan;
use std::sync::Arc;

fn try_as_cudf<T: ExecutionPlan + 'static>(
    r: datafusion::common::Result<T>,
) -> datafusion::common::Result<Option<Arc<dyn ExecutionPlan>>> {
    match r {
        Ok(n) => Ok(Some(Arc::new(n))),
        Err(DataFusionError::NotImplemented(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

#[derive(Debug)]
pub struct HostToCuDFRule;

impl PhysicalOptimizerRule for HostToCuDFRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let Some(cudf_config) = config.extensions.get::<CuDFConfig>() else {
            return Ok(plan);
        };
        if !cudf_config.enable {
            return Ok(plan);
        }

        let result = plan.transform_up(|mut plan| {
            let mut cudf_node: Option<Arc<dyn ExecutionPlan>> = None;
            if let Some(node) = plan.as_any().downcast_ref::<FilterExec>() {
                cudf_node = try_as_cudf(CuDFFilterExec::try_new(node.clone()))?;
            }

            if let Some(node) = plan.as_any().downcast_ref::<ProjectionExec>() {
                cudf_node = try_as_cudf(CuDFProjectionExec::try_new(node.clone()))?;
            }

            if let Some(node) = plan.as_any().downcast_ref::<SortExec>() {
                cudf_node = Some(Arc::new(CuDFSortExec::try_new(node.clone())?));
            }

            if let Some(node) = plan.as_any().downcast_ref::<CoalesceBatchesExec>() {
                if is_cudf_plan(node.input().as_ref()) {
                    cudf_node = Some(Arc::new(CuDFCoalesceBatchesExec::from_input(
                        node.input().clone(),
                        cudf_config.batch_size,
                    )));
                }
            }

            if let Some(node) = plan.as_any().downcast_ref::<HashJoinExec>() {
                cudf_node = try_as_cudf_hash_join(node)?;
            }

            if let Some(node) = plan.as_any().downcast_ref::<AggregateExec>() {
                cudf_node = try_as_cudf_aggregate(node)?;
            }

            let mut changed = false;
            if let Some(node) = cudf_node {
                plan = node;
                changed = true;
            }

            let plan_is_cudf = is_cudf_plan(plan.as_ref());
            let children = plan.children();
            let mut new_children: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(children.len());
            for child in children.iter() {
                let child_is_cudf = is_cudf_plan(child.as_ref());

                if plan_is_cudf && !child_is_cudf && !plan.as_any().is::<CuDFLoadExec>() {
                    if !child.as_any().is::<CoalesceBatchesExec>() {
                        let child = Arc::new(CoalesceBatchesExec::new(
                            Arc::clone(child),
                            cudf_config.batch_size,
                        ));
                        new_children.push(Arc::new(CuDFLoadExec::try_new(child)?));
                    } else {
                        new_children.push(Arc::new(CuDFLoadExec::try_new(Arc::clone(child))?));
                    }
                    changed = true;
                    continue;
                }

                if !plan_is_cudf && child_is_cudf && !child.as_any().is::<CuDFUnloadExec>() {
                    let mut unload = if !child.as_any().is::<CuDFCoalesceBatchesExec>() {
                        let child = Arc::new(CuDFCoalesceBatchesExec::from_input(
                            Arc::clone(child),
                            cudf_config.batch_size,
                        ));
                        CuDFUnloadExec::new(child)
                    } else {
                        CuDFUnloadExec::new(Arc::clone(child))
                    };
                    // Aggregations will expect a specific schema in, which is the one that was
                    // established while the node was placed there. As we are dealing with type
                    // incompatibilities in CuDF, we are tweaking the schema we return, and
                    // therefore, we might need to manually force a cast.
                    if let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() {
                        unload = unload.with_target_schema(Arc::clone(&agg.input_schema))
                    }
                    new_children.push(Arc::new(unload));
                    changed = true;
                    continue;
                }

                new_children.push(Arc::clone(child));
            }

            if changed {
                Ok(Transformed::yes(plan.with_new_children(new_children)?))
            } else {
                Ok(Transformed::no(plan))
            }
        })?;

        let plan = if is_cudf_plan(result.data.as_ref()) {
            Arc::new(CuDFUnloadExec::new(result.data))
        } else {
            result.data
        };

        assign_gpu_segment_ids(plan)
    }

    fn name(&self) -> &str {
        "HostToCuDFRule"
    }

    fn schema_check(&self) -> bool {
        false
    }
}

fn assign_gpu_segment_ids(
    plan: Arc<dyn ExecutionPlan>,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let mut next_segment_id = 0;
    Ok(assign_gpu_segment_ids_inner(plan, &mut next_segment_id)?.0)
}

fn assign_gpu_segment_ids_inner(
    plan: Arc<dyn ExecutionPlan>,
    next_segment_id: &mut usize,
) -> datafusion::common::Result<(Arc<dyn ExecutionPlan>, Option<usize>)> {
    let children: Vec<_> = plan.children().into_iter().cloned().collect();
    let mut child_segments = Vec::with_capacity(children.len());
    let mut new_children = Vec::with_capacity(children.len());
    let mut children_changed = false;

    for child in &children {
        let (new_child, segment_id) =
            assign_gpu_segment_ids_inner(Arc::clone(child), next_segment_id)?;
        children_changed |= !Arc::ptr_eq(child, &new_child);
        child_segments.push(segment_id);
        new_children.push(new_child);
    }

    let plan = if children_changed {
        plan.with_new_children(new_children)?
    } else {
        plan
    };

    if let Some(load) = plan.as_any().downcast_ref::<CuDFLoadExec>() {
        let segment_id = *next_segment_id;
        *next_segment_id += 1;
        return Ok((Arc::new(load.with_segment_id(segment_id)), Some(segment_id)));
    }

    if let Some(aggregate) = plan
        .as_any()
        .downcast_ref::<crate::aggregate::CuDFAggregateExec>()
    {
        let segment_id = child_segments
            .iter()
            .flatten()
            .next()
            .copied()
            .unwrap_or_else(|| aggregate.segment_id());
        return Ok((
            Arc::new(aggregate.with_segment_id(segment_id)),
            Some(segment_id),
        ));
    }

    if let Some(unload) = plan.as_any().downcast_ref::<CuDFUnloadExec>() {
        let segment_id = child_segments
            .iter()
            .flatten()
            .next()
            .copied()
            .unwrap_or_else(|| unload.segment_id());
        return Ok((
            Arc::new(unload.with_segment_id(segment_id)),
            Some(segment_id),
        ));
    }

    if is_cudf_plan(plan.as_ref()) {
        let segment_id = child_segments.iter().flatten().next().copied();
        return Ok((plan, segment_id));
    }

    Ok((plan, None))
}
