use crate::expr::expr_to_cudf_expr;
use crate::task_context::{cuda_streams_enabled, CuDFTaskContext};
use arrow_schema::Schema;
use datafusion::error::DataFusionError;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion_physical_plan::udaf::AggregateFunctionExpr;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, InputOrderMode, PlanProperties,
};
use std::any::{type_name, Any};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

mod op;
mod stream;

pub use op::avg::avg;
pub use op::count::count;
pub use op::max::max;
pub use op::min::min;
pub use op::sum::sum;
pub(crate) use op::udf::CuDFAggregateUDF;
pub(crate) use op::CuDFAggregationOp;

/// GPU-accelerated GROUP BY aggregate execution node.
///
/// Replaces DataFusion's `AggregateExec` for queries where all aggregate
/// functions have cuDF implementations.
#[derive(Debug)]
pub struct CuDFAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    segment_id: usize,
    mode: AggregateMode,
    group_by: PhysicalGroupBy,
    aggr_expr: Vec<Arc<AggregateFunctionExpr>>,

    plan_properties: PlanProperties,
}

impl CuDFAggregateExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        mode: AggregateMode,
        group_by: PhysicalGroupBy,
        aggr_expr: Vec<Arc<AggregateFunctionExpr>>,
    ) -> Result<Self> {
        Self::try_new_with_segment_id(input, 0, mode, group_by, aggr_expr)
    }

    pub fn try_new_with_segment_id(
        input: Arc<dyn ExecutionPlan>,
        segment_id: usize,
        mode: AggregateMode,
        group_by: PhysicalGroupBy,
        aggr_expr: Vec<Arc<AggregateFunctionExpr>>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        // Non-single grouping sets (CUBE, ROLLUP) add an extra column for the grouping ID.
        let group_by_fields = {
            let num_exprs = group_by.expr().len();
            if !group_by.is_single() {
                num_exprs + 1
            } else {
                num_exprs
            }
        };

        let group_by_schema = group_by.group_schema(&input_schema)?;
        let group_by_exprs = group_by_schema.fields.iter().take(group_by_fields).cloned();

        let mut fields = Vec::with_capacity(group_by_fields + aggr_expr.len());

        fields.extend(group_by_exprs);

        // Partial mode emits intermediate state columns (e.g., AVG emits [count, sum]).
        // All other modes emit the final result column (e.g., AVG emits [avg]).
        if mode == AggregateMode::Partial {
            for expr in &aggr_expr {
                for field in expr.state_fields()? {
                    fields.push(field);
                }
            }
        } else {
            for expr in &aggr_expr {
                fields.push(expr.field());
            }
        }

        let output_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata.clone(),
        ));

        let group_by_expr_mapping =
            ProjectionMapping::try_new(group_by.expr().iter().cloned(), &input.schema())?;

        let plan_properties = AggregateExec::compute_properties(
            &input,
            output_schema,
            &group_by_expr_mapping,
            &mode,
            &InputOrderMode::Linear,
            &aggr_expr,
        )?;

        Ok(Self {
            input,
            segment_id,
            mode,
            group_by,
            aggr_expr,
            plan_properties,
        })
    }

    pub fn segment_id(&self) -> usize {
        self.segment_id
    }

    pub(crate) fn with_segment_id(&self, segment_id: usize) -> Self {
        Self {
            input: Arc::clone(&self.input),
            segment_id,
            mode: self.mode,
            group_by: self.group_by.clone(),
            aggr_expr: self.aggr_expr.clone(),
            plan_properties: self.plan_properties.clone(),
        }
    }
}

impl DisplayAs for CuDFAggregateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CuDFAggregateExec: ")?;
        write!(f, "mode={:?}, ", self.mode)?;
        write!(f, "group_by=[")?;
        for (i, (expr, alias)) in self.group_by.expr().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}@{}", alias, expr)?;
        }
        write!(f, "], aggr_expr=[")?;
        for (i, expr) in self.aggr_expr.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", expr.name())?;
        }
        write!(f, "]")
    }
}

impl ExecutionPlan for CuDFAggregateExec {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let new = Self::try_new(
            children[0].clone(),
            self.mode,
            self.group_by.clone(),
            self.aggr_expr.clone(),
        )?
        .with_segment_id(self.segment_id);

        Ok(Arc::new(new))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let cuda_stream = if cuda_streams_enabled(&context) {
            let cudf_ctx = CuDFTaskContext::from_ctx(&context)?;
            Some(cudf_ctx.stream(self.segment_id, partition).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "CUDA stream not assigned for cuDF segment {} partition {}",
                    self.segment_id, partition
                ))
            })?)
        } else {
            None
        };
        let stream = stream::Stream::new(
            input,
            self.schema(),
            self.mode,
            self.group_by.clone(),
            self.aggr_expr.clone(),
            cuda_stream,
        )?;
        Ok(Box::pin(stream))
    }
}

/// Try to convert an `AggregateExec` to a `CuDFAggregateExec`.
///
/// Returns `Ok(None)` (CPU fallback) if any unsupported feature is detected:
/// - No GROUP BY columns (global aggregation requires synthetic key, not yet supported)
/// - Non-single grouping sets (CUBE, ROLLUP)
/// - DISTINCT or ORDER BY in any aggregate function
/// - Any aggregate function not backed by `CuDFAggregateUDF`
///
/// Note: GROUP BY keys with Arrow `Utf8View` (StringView) type are handled transparently.
/// `CuDFLoadExec` coerces `Utf8View -> Utf8` in both schema and data, and `CuDFUnloadExec`
/// casts back to `Utf8View` so upstream CPU nodes see the original type.
pub fn try_as_cudf_aggregate(node: &AggregateExec) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    // TODO: support global aggregation (no GROUP BY) by injecting a synthetic constant key.
    if node.group_expr().expr().is_empty() {
        return Ok(None);
    }
    // TODO: support CUBE and ROLLUP grouping sets.
    if !node.group_expr().is_single() {
        return Ok(None);
    }
    for expr in node.aggr_expr() {
        // TODO: support DISTINCT aggregates (e.g. COUNT DISTINCT).
        // TODO: support ORDER BY inside aggregate functions (e.g. ARRAY_AGG(x ORDER BY x)).
        if expr.is_distinct() || !expr.order_bys().is_empty() {
            return Ok(None);
        }
        if expr
            .fun()
            .inner()
            .as_any()
            .downcast_ref::<CuDFAggregateUDF>()
            .is_none()
        {
            return Ok(None);
        }
        // Validate that each input expression is cuDF-compatible.
        for input_expr in expr.expressions() {
            if expr_to_cudf_expr(input_expr.as_ref()).is_err() {
                return Ok(None);
            }
        }
    }
    Ok(Some(Arc::new(CuDFAggregateExec::try_new(
        node.input().clone(),
        *node.mode(),
        node.group_expr().clone(),
        node.aggr_expr().to_vec(),
    )?)))
}

#[cfg(test)]
mod test {
    use crate::aggregate::op::avg::avg;
    use crate::aggregate::op::count::count;
    use crate::aggregate::op::max::max;
    use crate::aggregate::op::min::min;
    use crate::aggregate::op::sum::sum;
    use crate::aggregate::CuDFAggregateExec;
    use crate::assert_snapshot;
    use crate::physical::{CuDFLoadExec, CuDFUnloadExec};
    use crate::{CuDFConfig, CuDFExt};
    use arrow::array::record_batch;
    use arrow::util::pretty::pretty_format_batches;
    use arrow_schema::SchemaRef;
    use datafusion::common::ScalarValue;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::execution::TaskContext;
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    use datafusion::prelude::SessionConfig;
    use datafusion_expr::AggregateUDF;
    use datafusion_physical_plan::aggregates::{AggregateMode, PhysicalGroupBy};
    use datafusion_physical_plan::expressions::{col, Literal};
    use datafusion_physical_plan::test::TestMemoryExec;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::PhysicalExpr;
    use futures_util::TryStreamExt;
    use std::collections::HashMap;
    use std::error::Error;
    use std::sync::Arc;

    /// Run a GROUP BY aggregation through the full GPU pipeline:
    /// TestMemoryExec -> CuDFLoadExec -> CuDFAggregateExec -> CuDFUnloadExec.
    ///
    /// Sends 3 identical batches (to exercise cross-batch rolling merge) and
    /// groups by column "c". `build_args` receives the batch schema and returns
    /// the argument expressions for the aggregate function.
    async fn run_group_by_test(
        agg_fn: Arc<AggregateUDF>,
        build_args: impl FnOnce(&SchemaRef) -> datafusion::error::Result<Vec<Arc<dyn PhysicalExpr>>>,
        agg_alias: &str,
    ) -> Result<String, Box<dyn Error>> {
        run_group_by_test_with_cuda_streams(agg_fn, build_args, agg_alias, false).await
    }

    async fn run_group_by_test_with_cuda_streams(
        agg_fn: Arc<AggregateUDF>,
        build_args: impl FnOnce(&SchemaRef) -> datafusion::error::Result<Vec<Arc<dyn PhysicalExpr>>>,
        agg_alias: &str,
        cuda_streams: bool,
    ) -> Result<String, Box<dyn Error>> {
        let batch = record_batch!(
            ("a", Int64, [1, 4, 3]),
            ("b", Float64, [Some(4.0), None, Some(5.0)]),
            ("c", Utf8, ["hello", "hello", "world"]),
            ("d", Float64, [4.0, 5.0, 5.0])
        )
        .expect("created batch");

        let schema = batch.schema();

        let root = TestMemoryExec::try_new(
            &[vec![batch.clone(), batch.clone(), batch]],
            schema.clone(),
            None,
        )?;
        let load = CuDFLoadExec::try_new(Arc::new(root))?;

        let group_by = PhysicalGroupBy::new_single(vec![(col("c", &schema)?, "c".to_string())]);

        let agg = AggregateExprBuilder::new(agg_fn, build_args(&schema)?)
            .schema(schema)
            .alias(agg_alias)
            .build()?;

        let aggregate = CuDFAggregateExec::try_new(
            Arc::new(load),
            AggregateMode::Single,
            group_by,
            vec![Arc::new(agg)],
        )?;

        let unload = CuDFUnloadExec::new(Arc::new(aggregate));

        let task = if cuda_streams {
            let mut cudf_config = CuDFConfig::default();
            cudf_config.cuda_streams = true;
            Arc::new(
                make_task_context(SessionConfig::new().with_option_extension(cudf_config))
                    .with_cudf_task_context(),
            )
        } else {
            Arc::new(TaskContext::default())
        };

        let result = unload.execute(0, task)?;
        let batches = result.try_collect::<Vec<_>>().await?;

        let output = pretty_format_batches(&batches)?.to_string();
        Ok(output)
    }

    fn make_task_context(config: SessionConfig) -> TaskContext {
        TaskContext::new(
            Some("task_id".to_string()),
            "session_id".to_string(),
            config,
            HashMap::default(),
            HashMap::default(),
            HashMap::default(),
            Arc::new(RuntimeEnv::default()),
        )
    }

    #[tokio::test]
    async fn test_group_by_sum() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(sum(), |s| Ok(vec![col("a", s)?]), "SUM(a)").await?;

        // Note: cuDF's SUM always returns Int64 for integer inputs
        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | SUM(a) |
        +-------+--------+
        | world | 9      |
        | hello | 15     |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_sum_cuda_streams_enabled() -> Result<(), Box<dyn Error>> {
        let output =
            run_group_by_test_with_cuda_streams(sum(), |s| Ok(vec![col("a", s)?]), "SUM(a)", true)
                .await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | SUM(a) |
        +-------+--------+
        | world | 9      |
        | hello | 15     |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_min() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(min(), |s| Ok(vec![col("a", s)?]), "MIN(a)").await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | MIN(a) |
        +-------+--------+
        | world | 3      |
        | hello | 1      |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_max() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(max(), |s| Ok(vec![col("a", s)?]), "MAX(a)").await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | MAX(a) |
        +-------+--------+
        | world | 3      |
        | hello | 4      |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_count() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(count(), |s| Ok(vec![col("a", s)?]), "COUNT(a)").await?;

        assert_snapshot!(output, @r"
        +-------+----------+
        | c     | COUNT(a) |
        +-------+----------+
        | world | 3        |
        | hello | 6        |
        +-------+----------+
        ");

        Ok(())
    }

    /// COUNT with a literal argument — the exact code path hit by COUNT(*) = COUNT(lit(1)).
    #[tokio::test]
    async fn test_group_by_count_literal_arg() -> Result<(), Box<dyn Error>> {
        let lit_one: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Int32(Some(1))));
        let output = run_group_by_test(count(), |_| Ok(vec![lit_one.clone()]), "COUNT(*)").await?;

        assert_snapshot!(output, @r"
        +-------+----------+
        | c     | COUNT(*) |
        +-------+----------+
        | world | 3        |
        | hello | 6        |
        +-------+----------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_avg() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(avg(), |s| Ok(vec![col("a", s)?]), "AVG(a)").await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | AVG(a) |
        +-------+--------+
        | world | 3.0    |
        | hello | 2.5    |
        +-------+--------+
        ");

        Ok(())
    }
}

/// Integration tests: full SQL pipeline through TestFramework against real weather data.
///
/// Tests using `check_query_results` omit ORDER BY — the helper sorts rows before
/// comparing, keeping plans free of sort operators. Float tests (AVG, mixed aggregates)
/// keep ORDER BY and use `assert_batches_approx_eq` to absorb last-ULP differences.
#[cfg(test)]
mod integration {
    use crate::aggregate::CuDFAggregateExec;
    use crate::assert_snapshot;
    use crate::physical::{CuDFLoadExec, CuDFUnloadExec};
    use crate::test_utils::{check_query_results, sort_batches, TestFramework};
    use arrow::array::{Array, Float64Array, RecordBatch};
    use arrow::util::pretty::pretty_format_batches;
    use datafusion_physical_plan::ExecutionPlan;
    use std::error::Error;

    /// Absorbs last-ULP differences between cuDF and DataFusion float arithmetic.
    /// Used only for tests that produce Float64 results (AVG, mixed aggregates).
    fn assert_batches_approx_eq(gpu: &[RecordBatch], cpu: &[RecordBatch], decimals: u32) {
        let factor = 10f64.powi(decimals as i32);
        assert_eq!(gpu.len(), cpu.len(), "batch count mismatch");
        for (b, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g.num_rows(), c.num_rows(), "batch {b}: row count mismatch");
            assert_eq!(g.num_columns(), c.num_columns(), "batch {b}: column count");
            for col in 0..g.num_columns() {
                let gc = g.column(col);
                let cc = c.column(col);
                if let (Some(gf), Some(cf)) = (
                    gc.as_any().downcast_ref::<Float64Array>(),
                    cc.as_any().downcast_ref::<Float64Array>(),
                ) {
                    for row in 0..gf.len() {
                        let gv = (gf.value(row) * factor).round() / factor;
                        let cv = (cf.value(row) * factor).round() / factor;
                        assert_eq!(gv, cv, "batch {b}, col {col}, row {row}");
                    }
                } else {
                    assert_eq!(gc.as_ref(), cc.as_ref(), "batch {b}, col {col}");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sum() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", SUM("Rainfall") as total_rain FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 1).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, sum(weather.Rainfall)@1 as total_rain]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[sum(weather.Rainfall)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@1], aggr_expr=[sum(weather.Rainfall)]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[Rainfall, RainToday], file_type=parquet
        ");
        Ok(())
    }

    #[tokio::test]
    async fn test_count() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT("Rainfall") as n FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 1).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, count(weather.Rainfall)@1 as n]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[count(weather.Rainfall)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@1], aggr_expr=[count(weather.Rainfall)]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[Rainfall, RainToday], file_type=parquet
        ");
        Ok(())
    }

    /// AVG produces Float64 — uses assert_batches_approx_eq to handle last-ULP differences.
    #[tokio::test]
    async fn test_avg() -> Result<(), Box<dyn Error>> {
        let sql =
            r#"SELECT "RainToday", AVG("MinTemp") as avg_min FROM weather GROUP BY "RainToday""#;
        let tf = TestFramework::new().await;
        let gpu = tf
            .execute(&format!(
                "SET cudf.enable=true; SET datafusion.execution.target_partitions=1; {sql}"
            ))
            .await?;
        let cpu = tf
            .execute(&format!(
                "SET datafusion.execution.target_partitions=1; {sql}"
            ))
            .await?;
        assert_batches_approx_eq(&sort_batches(&gpu.batches), &sort_batches(&cpu.batches), 10);
        assert_snapshot!(gpu.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, avg(weather.MinTemp)@1 as avg_min]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[avg(weather.MinTemp)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@1], aggr_expr=[avg(weather.MinTemp)]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[MinTemp, RainToday], file_type=parquet
        ");
        Ok(())
    }

    #[tokio::test]
    async fn test_min_max() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", MIN("MinTemp") as lo, MAX("MaxTemp") as hi FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 1).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, min(weather.MinTemp)@1 as lo, max(weather.MaxTemp)@2 as hi]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[min(weather.MinTemp), max(weather.MaxTemp)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@2], aggr_expr=[min(weather.MinTemp), max(weather.MaxTemp)]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[MinTemp, MaxTemp, RainToday], file_type=parquet
        ");
        Ok(())
    }

    /// Contains AVG — uses assert_batches_approx_eq to handle last-ULP differences.
    #[tokio::test]
    async fn test_multiple_aggregates() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT("Rainfall") as n, SUM("Rainfall") as total, AVG("MaxTemp") as avg_max, MIN("MinTemp") as lo, MAX("MaxTemp") as hi FROM weather GROUP BY "RainToday""#;
        let tf = TestFramework::new().await;
        let gpu = tf
            .execute(&format!(
                "SET cudf.enable=true; SET datafusion.execution.target_partitions=1; {sql}"
            ))
            .await?;
        let cpu = tf
            .execute(&format!(
                "SET datafusion.execution.target_partitions=1; {sql}"
            ))
            .await?;
        assert_batches_approx_eq(&sort_batches(&gpu.batches), &sort_batches(&cpu.batches), 10);
        assert_snapshot!(gpu.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, count(weather.Rainfall)@1 as n, sum(weather.Rainfall)@2 as total, avg(weather.MaxTemp)@3 as avg_max, min(weather.MinTemp)@4 as lo, max(weather.MaxTemp)@5 as hi]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[count(weather.Rainfall), sum(weather.Rainfall), avg(weather.MaxTemp), min(weather.MinTemp), max(weather.MaxTemp)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@3], aggr_expr=[count(weather.Rainfall), sum(weather.Rainfall), avg(weather.MaxTemp), min(weather.MinTemp), max(weather.MaxTemp)]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[MinTemp, MaxTemp, Rainfall, RainToday], file_type=parquet
        ");
        Ok(())
    }

    #[tokio::test]
    async fn test_count_star() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT(*) as n FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 1).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, count(Int64(1))@1 as n]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[count(Int64(1))]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@0], aggr_expr=[count(Int64(1))]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
        ");
        Ok(())
    }

    #[tokio::test]
    async fn test_count_star_mixed() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT(*) as n, SUM("Rainfall") as total FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 1).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, count(Int64(1))@1 as n, sum(weather.Rainfall)@2 as total]
            CuDFAggregateExec: mode=Final, group_by=[RainToday@RainToday@0], aggr_expr=[count(Int64(1)), sum(weather.Rainfall)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=81920
                  CoalescePartitionsExec
                    CuDFUnloadExec
                      CuDFCoalesceBatchesExec: target_batch_size=81920
                        CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@1], aggr_expr=[count(Int64(1)), sum(weather.Rainfall)]
                          CuDFLoadExec
                            CoalesceBatchesExec: target_batch_size=81920
                              DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[Rainfall, RainToday], file_type=parquet
        ");
        Ok(())
    }

    #[tokio::test]
    async fn test_multi_partition_sum() -> Result<(), Box<dyn Error>> {
        let sql =
            r#"SELECT "RainToday", SUM("Rainfall") as total FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 4).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, sum(weather.Rainfall)@1 as total]
            CuDFAggregateExec: mode=FinalPartitioned, group_by=[RainToday@RainToday@0], aggr_expr=[sum(weather.Rainfall)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=8192
                  RepartitionExec: partitioning=Hash([RainToday@0], 4), input_partitions=4
                    RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3
                      CuDFUnloadExec
                        CuDFCoalesceBatchesExec: target_batch_size=81920
                          CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@1], aggr_expr=[sum(weather.Rainfall)]
                            CuDFLoadExec
                              CoalesceBatchesExec: target_batch_size=81920
                                DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[Rainfall, RainToday], file_type=parquet
        ");
        Ok(())
    }

    #[tokio::test]
    async fn test_multi_partition_count_cuda_streams() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT(*) as n FROM weather GROUP BY "RainToday""#;
        let gpu_tf = TestFramework::new().await;
        let cpu_tf = TestFramework::new().await;

        let gpu = gpu_tf
            .execute(&format!(
                "SET cudf.enable=true; SET cudf.cuda_streams=true; SET datafusion.execution.target_partitions=4; {sql}"
            ))
            .await?;
        let cpu = cpu_tf
            .execute(&format!(
                "SET datafusion.execution.target_partitions=4; {sql}"
            ))
            .await?;

        let gpu_pp = pretty_format_batches(&sort_batches(&gpu.batches))?.to_string();
        let cpu_pp = pretty_format_batches(&sort_batches(&cpu.batches))?.to_string();
        assert_eq!(gpu_pp, cpu_pp);
        Ok(())
    }

    #[tokio::test]
    async fn test_repartitioned_aggregate_segments_are_distinct() -> Result<(), Box<dyn Error>> {
        let sql =
            r#"SELECT "RainToday", SUM("Rainfall") as total FROM weather GROUP BY "RainToday""#;
        let tf = TestFramework::new().await;
        let plan = tf
            .plan(&format!(
                "SET cudf.enable=true; SET datafusion.execution.target_partitions=4; {sql}"
            ))
            .await?;

        let mut segments = PlanSegments::default();
        collect_plan_segments(plan.plan.as_ref(), &mut segments);
        segments.sort_dedup();

        assert_eq!(segments.loads, vec![0, 1]);
        assert_eq!(segments.aggregates, vec![0, 1]);
        assert_eq!(segments.unloads, vec![0, 1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_multi_partition_count_star() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT(*) as n FROM weather GROUP BY "RainToday""#;
        let result = check_query_results(sql, 4).await?;
        assert_snapshot!(result.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, count(Int64(1))@1 as n]
            CuDFAggregateExec: mode=FinalPartitioned, group_by=[RainToday@RainToday@0], aggr_expr=[count(Int64(1))]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=8192
                  RepartitionExec: partitioning=Hash([RainToday@0], 4), input_partitions=4
                    RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3
                      CuDFUnloadExec
                        CuDFCoalesceBatchesExec: target_batch_size=81920
                          CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@0], aggr_expr=[count(Int64(1))]
                            CuDFLoadExec
                              CoalesceBatchesExec: target_batch_size=81920
                                DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
        ");
        Ok(())
    }

    /// Contains AVG — uses assert_batches_approx_eq to handle last-ULP differences.
    #[tokio::test]
    async fn test_multi_partition_multiple_aggs() -> Result<(), Box<dyn Error>> {
        let sql = r#"SELECT "RainToday", COUNT(*) as n, SUM("Rainfall") as total, AVG("MaxTemp") as avg_max, MIN("MinTemp") as lo, MAX("MaxTemp") as hi FROM weather GROUP BY "RainToday""#;
        let tf = TestFramework::new().await;
        let gpu = tf
            .execute(&format!(
                "SET cudf.enable=true; SET datafusion.execution.target_partitions=4; {sql}"
            ))
            .await?;
        let cpu = tf
            .execute(&format!(
                "SET datafusion.execution.target_partitions=4; {sql}"
            ))
            .await?;
        assert_batches_approx_eq(&sort_batches(&gpu.batches), &sort_batches(&cpu.batches), 10);
        assert_snapshot!(gpu.plan, @"
        CuDFUnloadExec
          CuDFProjectionExec: expr=[RainToday@0 as RainToday, count(Int64(1))@1 as n, sum(weather.Rainfall)@2 as total, avg(weather.MaxTemp)@3 as avg_max, min(weather.MinTemp)@4 as lo, max(weather.MaxTemp)@5 as hi]
            CuDFAggregateExec: mode=FinalPartitioned, group_by=[RainToday@RainToday@0], aggr_expr=[count(Int64(1)), sum(weather.Rainfall), avg(weather.MaxTemp), min(weather.MinTemp), max(weather.MaxTemp)]
              CuDFLoadExec
                CoalesceBatchesExec: target_batch_size=8192
                  RepartitionExec: partitioning=Hash([RainToday@0], 4), input_partitions=4
                    RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3
                      CuDFUnloadExec
                        CuDFCoalesceBatchesExec: target_batch_size=81920
                          CuDFAggregateExec: mode=Partial, group_by=[RainToday@RainToday@3], aggr_expr=[count(Int64(1)), sum(weather.Rainfall), avg(weather.MaxTemp), min(weather.MinTemp), max(weather.MaxTemp)]
                            CuDFLoadExec
                              CoalesceBatchesExec: target_batch_size=81920
                                DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[MinTemp, MaxTemp, Rainfall, RainToday], file_type=parquet
        ");
        Ok(())
    }

    /// Aggregates with unsupported functions (non-CuDFAggregateUDF) must fall back to CPU.
    #[tokio::test]
    async fn test_unsupported_agg_falls_back_to_cpu() -> Result<(), Box<dyn Error>> {
        let tf = TestFramework::new().await;
        let sql = r#"SELECT "RainToday", BOOL_OR("RainTomorrow" = 'Yes') as any_rain FROM weather GROUP BY "RainToday""#;
        let gpu = tf.execute(&format!("SET cudf.enable=true; {sql}")).await?;
        let cpu = tf.execute(sql).await?;
        assert!(
            !gpu.plan.contains("CuDFAggregateExec"),
            "expected CPU fallback"
        );
        assert_eq!(cpu.pretty_print, gpu.pretty_print);
        Ok(())
    }

    #[derive(Default)]
    struct PlanSegments {
        loads: Vec<usize>,
        aggregates: Vec<usize>,
        unloads: Vec<usize>,
    }

    impl PlanSegments {
        fn sort_dedup(&mut self) {
            self.loads.sort_unstable();
            self.loads.dedup();
            self.aggregates.sort_unstable();
            self.aggregates.dedup();
            self.unloads.sort_unstable();
            self.unloads.dedup();
        }
    }

    fn collect_plan_segments(plan: &dyn ExecutionPlan, segments: &mut PlanSegments) {
        if let Some(load) = plan.as_any().downcast_ref::<CuDFLoadExec>() {
            segments.loads.push(load.segment_id());
        }
        if let Some(aggregate) = plan.as_any().downcast_ref::<CuDFAggregateExec>() {
            segments.aggregates.push(aggregate.segment_id());
        }
        if let Some(unload) = plan.as_any().downcast_ref::<CuDFUnloadExec>() {
            segments.unloads.push(unload.segment_id());
        }
        for child in plan.children() {
            collect_plan_segments(child.as_ref(), segments);
        }
    }
}
