use crate::errors::cudf_to_df;
use crate::optimizer::CuDFConfig;
use crate::task_context::{cuda_streams_enabled, CuDFTaskContext};
use arrow::array::{Array, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::{exec_err, plan_err, ScalarValue};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures_util::stream::StreamExt;
use libcudf_rs::{is_cudf_array, CuDFTable};
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

#[derive(Debug)]
pub struct CuDFLoadExec {
    input: Arc<dyn ExecutionPlan>,
    segment_id: usize,

    properties: PlanProperties,
}

impl CuDFLoadExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self, DataFusionError> {
        Self::try_new_with_segment_id(input, 0)
    }

    pub fn try_new_with_segment_id(
        input: Arc<dyn ExecutionPlan>,
        segment_id: usize,
    ) -> Result<Self, DataFusionError> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(cudf_schema_compatibility_map(input.schema())),
            input.properties().partitioning.clone(),
            input.properties().emission_type,
            input.properties().boundedness,
        );
        Ok(Self {
            input,
            segment_id,
            properties,
        })
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

impl DisplayAs for CuDFLoadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CuDFLoadExec")
    }
}

impl ExecutionPlan for CuDFLoadExec {
    fn name(&self) -> &str {
        "CuDFLoadExec"
    }

    // TODO: implement partition_statistics by delegating to self.input.partition_statistics().
    // The schema changes (cudf_schema_compatibility_map remaps types), but row counts and
    // column-level statistics remain valid and should be preserved.

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
                "CuDFLoadExec expects exactly 1 child, {} where provided",
                children.len()
            );
        }
        let input = Arc::clone(&children[0]);
        Ok(Arc::new(Self::try_new_with_segment_id(
            input,
            self.segment_id,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let cuda_stream = if cuda_streams_enabled(&context) {
            let cudf_cfg = CuDFConfig::from_config_options(context.session_config().options())?;
            let cudf_ctx = CuDFTaskContext::from_ctx(&context)?;
            let stream = cudf_ctx
                .stream(self.segment_id, partition)
                .unwrap_or_else(|| {
                    let stream = cudf_cfg.stream_source().allocate();
                    cudf_ctx.set_stream(self.segment_id, partition, Arc::clone(&stream));
                    stream
                });
            Some(stream)
        } else {
            None
        };

        let host_stream = self.input.execute(partition, context)?;
        let target_schema = self.schema();

        let cudf_stream = host_stream.map(move |batch_or_err| {
            let batch = match batch_or_err {
                Ok(batch) => cast_to_target_schema(batch, Arc::clone(&target_schema))?,
                Err(err) => return Err(err),
            };

            if batch.columns().iter().any(|c| is_cudf_array(c)) {
                return exec_err!(
                    "Cannot move RecordBatch from host to CuDF: a column is already a CuDF array"
                );
            }
            let schema = batch.schema();
            let table = match cuda_stream.as_deref() {
                Some(stream) => CuDFTable::from_arrow_host_on(batch, stream).map_err(cudf_to_df)?,
                None => CuDFTable::from_arrow_host(batch).map_err(cudf_to_df)?,
            };
            let cudf_cols: Vec<Arc<dyn Array>> = table
                .into_columns()
                .into_iter()
                .map(|c| Arc::new(c.into_view()) as Arc<dyn Array>)
                .collect();
            Ok(libcudf_rs::record_batch_with_schema(cudf_cols, &schema)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            cudf_stream,
        )))
    }
}

/// Converts Arrow scalar types that cuDF doesn't support (e.g. `Utf8View`) into their
/// cuDF-compatible equivalents.
pub(crate) fn normalize_scalar_for_cudf(value: ScalarValue) -> ScalarValue {
    match value {
        ScalarValue::Utf8View(s) => ScalarValue::Utf8(s),
        other => other,
    }
}

/// Maps an Arrow schema to cuDF-compatible types (`Utf8View -> Utf8`).
pub(crate) fn cudf_schema_compatibility_map(schema: SchemaRef) -> SchemaRef {
    let mut new_fields = Vec::with_capacity(schema.fields.len());

    for field in schema.fields() {
        let field = match field.data_type() {
            DataType::Utf8View => FieldRef::new(Field::new(
                field.name(),
                DataType::Utf8,
                field.is_nullable(),
            )),
            _ => Arc::clone(field),
        };
        new_fields.push(field);
    }

    SchemaRef::new(Schema::new(new_fields))
}

pub(crate) fn cast_to_target_schema(
    batch: RecordBatch,
    target_schema: SchemaRef,
) -> Result<RecordBatch, ArrowError> {
    let columns = batch
        .columns()
        .iter()
        .zip(target_schema.fields())
        .map(|(col, field)| arrow::compute::cast(col, field.data_type()))
        .collect::<Result<Vec<_>, _>>()?;

    RecordBatch::try_new(target_schema, columns)
}
