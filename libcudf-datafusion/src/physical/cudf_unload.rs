use crate::errors::cudf_to_df;
use crate::task_context::{cuda_streams_enabled, CuDFTaskContext};
use arrow::array::RecordBatch;
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::{plan_err, DataFusionError};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures_util::StreamExt;
use libcudf_rs::CuDFTableView;
use std::any::Any;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

#[derive(Debug)]
pub struct CuDFUnloadExec {
    input: Arc<dyn ExecutionPlan>,
    segment_id: usize,
    properties: PlanProperties,
}

impl CuDFUnloadExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self::new_with_segment_id(input, 0)
    }

    pub fn new_with_segment_id(input: Arc<dyn ExecutionPlan>, segment_id: usize) -> Self {
        let mut properties = input.properties().clone();
        properties.eq_properties =
            EquivalenceProperties::new(cudf_unload_schema_map(input.schema()));
        Self {
            properties,
            input,
            segment_id,
        }
    }

    pub fn with_target_schema(&self, target_schema: SchemaRef) -> Self {
        let mut properties = self.properties.clone();
        properties.eq_properties = EquivalenceProperties::new(target_schema);
        Self {
            properties,
            input: Arc::clone(&self.input),
            segment_id: self.segment_id,
        }
    }

    pub fn segment_id(&self) -> usize {
        self.segment_id
    }

    pub(crate) fn with_segment_id(&self, segment_id: usize) -> Self {
        Self {
            input: Arc::clone(&self.input),
            segment_id,
            properties: self.properties.clone(),
        }
    }
}

impl DisplayAs for CuDFUnloadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CuDFUnloadExec")
    }
}

impl ExecutionPlan for CuDFUnloadExec {
    fn name(&self) -> &str {
        "CuDFUnloadExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!(
                "CuDFUnloadExec expects exactly 1 child, {} where provided",
                children.len()
            );
        }
        let input = Arc::clone(&children[0]);
        Ok(Arc::new(Self::new_with_segment_id(input, self.segment_id)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let cudf_stream = self.input.execute(partition, Arc::clone(&context))?;
        let (cuda_stream, cleanup) = if cuda_streams_enabled(&context) {
            let cudf_ctx = CuDFTaskContext::from_ctx(&context)?;
            let stream = cudf_ctx.stream(self.segment_id, partition).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "CUDA stream not assigned for cuDF segment {} partition {}",
                    self.segment_id, partition
                ))
            })?;
            (
                Some(stream),
                Some(StreamCleanup {
                    cudf_ctx,
                    segment_id: self.segment_id,
                    partition,
                }),
            )
        } else {
            (None, None)
        };
        let target_schema = self.schema();
        let host_stream = cudf_stream.map(move |batch_or_err| {
            let batch = match batch_or_err {
                Ok(batch) => batch,
                Err(err) => return Err(err),
            };

            let view = CuDFTableView::from_record_batch(&batch).map_err(cudf_to_df)?;
            let host_batch = match cuda_stream.as_deref() {
                Some(stream) => view.to_arrow_host_on(stream),
                None => view.to_arrow_host(),
            }
            .map_err(cudf_to_df)?;
            let columns = host_batch
                .columns()
                .iter()
                .zip(target_schema.fields())
                .map(|(col, field)| arrow::compute::cast(col, field.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new(target_schema.clone(), columns).map_err(|err| {
                DataFusionError::ArrowError(
                    Box::new(err),
                    Some("Error while unloading a RecordBatch from CuDF into host".to_string()),
                )
            })
        });
        let host_stream = UnsetStreamOnEnd::new(Box::pin(host_stream), cleanup);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            host_stream,
        )))
    }
}

struct StreamCleanup {
    cudf_ctx: Arc<CuDFTaskContext>,
    segment_id: usize,
    partition: usize,
}

struct UnsetStreamOnEnd {
    inner:
        Pin<Box<dyn futures_util::Stream<Item = datafusion::common::Result<RecordBatch>> + Send>>,
    cleanup: Option<StreamCleanup>,
}

impl UnsetStreamOnEnd {
    fn new(
        inner: Pin<
            Box<dyn futures_util::Stream<Item = datafusion::common::Result<RecordBatch>> + Send>,
        >,
        cleanup: Option<StreamCleanup>,
    ) -> Self {
        Self { inner, cleanup }
    }

    fn cleanup(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup
                .cudf_ctx
                .unset_stream(cleanup.segment_id, cleanup.partition);
        }
    }
}

impl futures_util::Stream for UnsetStreamOnEnd {
    type Item = datafusion::common::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) {
            self.cleanup();
        }
        poll
    }
}

impl Drop for UnsetStreamOnEnd {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Partial reverse of [`cudf_schema_compatibility_map`]: restores `Utf8 -> Utf8View` so
/// downstream CPU nodes see the types they expect from the original parquet schema.
///
/// Only Utf8View is reversed, decimal precision normalization is not because the original
/// precision is not recoverable.
fn cudf_unload_schema_map(schema: SchemaRef) -> SchemaRef {
    let new_fields: Vec<FieldRef> = schema
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::Utf8 => FieldRef::new(Field::new(
                field.name(),
                DataType::Utf8View,
                field.is_nullable(),
            )),
            _ => Arc::clone(field),
        })
        .collect();
    SchemaRef::new(Schema::new(new_fields))
}

#[cfg(test)]
mod tests {
    use super::CuDFUnloadExec;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_physical_plan::{test::TestMemoryExec, ExecutionPlan};
    use std::sync::Arc;

    #[test]
    fn test_schema_restores_utf8view() {
        // Input schema uses Utf8 (cuDF's normalised string type).
        // CuDFUnloadExec must restore Utf8View so downstream CPU nodes see the
        // type they expect from the original parquet schema.
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));
        let input = Arc::new(TestMemoryExec::try_new(&[], schema, None).unwrap());
        let unload = CuDFUnloadExec::new(input);
        let out = unload.schema();
        assert_eq!(out.field(0).data_type(), &DataType::Utf8View);
        assert_eq!(out.field(1).data_type(), &DataType::Int32);
    }
}
