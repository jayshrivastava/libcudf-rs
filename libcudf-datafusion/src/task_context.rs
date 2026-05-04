use crate::optimizer::CuDFConfig;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use libcudf_rs::CuDFStream;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Query-scoped cuDF runtime state attached to a [`TaskContext`].
///
/// This keeps the current CUDA stream for each active GPU segment and partition.
#[derive(Default)]
pub struct CuDFTaskContext {
    streams: Mutex<HashMap<(usize, usize), Arc<CuDFStream>>>,
}

impl CuDFTaskContext {
    pub fn from_ctx(ctx: &Arc<TaskContext>) -> Result<Arc<Self>, DataFusionError> {
        ctx.session_config().get_extension::<Self>().ok_or_else(|| {
            DataFusionError::Internal(
                "CuDFTaskContext extension not installed in TaskContext".into(),
            )
        })
    }

    pub fn stream(&self, segment_id: usize, partition: usize) -> Option<Arc<CuDFStream>> {
        self.streams
            .lock()
            .expect("CuDFTaskContext mutex poisoned")
            .get(&(segment_id, partition))
            .cloned()
    }

    pub fn set_stream(&self, segment_id: usize, partition: usize, stream: Arc<CuDFStream>) {
        self.streams
            .lock()
            .expect("CuDFTaskContext mutex poisoned")
            .insert((segment_id, partition), stream);
    }

    pub fn unset_stream(&self, segment_id: usize, partition: usize) -> Option<Arc<CuDFStream>> {
        self.streams
            .lock()
            .expect("CuDFTaskContext mutex poisoned")
            .remove(&(segment_id, partition))
    }
}

pub(crate) fn cuda_streams_enabled(ctx: &TaskContext) -> bool {
    ctx.session_config()
        .options()
        .extensions
        .get::<CuDFConfig>()
        .map_or(false, |cfg| cfg.cuda_streams)
}

#[cfg(test)]
mod tests {
    use super::CuDFTaskContext;
    use crate::CuDFExt;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::execution::TaskContext;
    use datafusion::prelude::SessionConfig;
    use libcudf_rs::{CuDFStream, CuDFStreamFlags};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_streams_can_be_set_and_unset() -> Result<(), Box<dyn std::error::Error>> {
        let stream = Arc::new(CuDFStream::with_flags(CuDFStreamFlags::NonBlocking));
        let task_ctx = CuDFTaskContext::default();

        assert!(task_ctx.stream(0, 0).is_none());
        task_ctx.set_stream(0, 0, Arc::clone(&stream));
        assert!(Arc::ptr_eq(&stream, &task_ctx.stream(0, 0).unwrap()));
        assert!(Arc::ptr_eq(&stream, &task_ctx.unset_stream(0, 0).unwrap()));
        assert!(task_ctx.stream(0, 0).is_none());
        Ok(())
    }

    #[test]
    fn test_streams_are_segment_and_partition_scoped() -> Result<(), Box<dyn std::error::Error>> {
        let task_ctx = CuDFTaskContext::default();
        let segment0_partition1 = Arc::new(CuDFStream::with_flags(CuDFStreamFlags::NonBlocking));
        let segment1_partition1 = Arc::new(CuDFStream::with_flags(CuDFStreamFlags::NonBlocking));
        let segment0_partition2 = Arc::new(CuDFStream::with_flags(CuDFStreamFlags::NonBlocking));

        task_ctx.set_stream(0, 1, Arc::clone(&segment0_partition1));
        task_ctx.set_stream(1, 1, Arc::clone(&segment1_partition1));
        task_ctx.set_stream(0, 2, Arc::clone(&segment0_partition2));

        assert!(Arc::ptr_eq(
            &segment0_partition1,
            &task_ctx.stream(0, 1).unwrap()
        ));
        assert!(Arc::ptr_eq(
            &segment1_partition1,
            &task_ctx.stream(1, 1).unwrap()
        ));
        assert!(Arc::ptr_eq(
            &segment0_partition2,
            &task_ctx.stream(0, 2).unwrap()
        ));

        task_ctx.unset_stream(0, 1);
        assert!(task_ctx.stream(0, 1).is_none());
        assert!(task_ctx.stream(1, 1).is_some());
        assert!(task_ctx.stream(0, 2).is_some());
        Ok(())
    }

    #[test]
    fn test_from_ctx_reads_installed_extension() -> Result<(), Box<dyn std::error::Error>> {
        let task_ctx = Arc::new(make_task_context(SessionConfig::new()).with_cudf_task_context());
        let cudf_ctx = CuDFTaskContext::from_ctx(&task_ctx)?;
        let stream = Arc::new(CuDFStream::with_flags(CuDFStreamFlags::NonBlocking));
        cudf_ctx.set_stream(0, 1, Arc::clone(&stream));
        assert!(Arc::ptr_eq(&stream, &cudf_ctx.stream(0, 1).unwrap()));
        Ok(())
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
}
