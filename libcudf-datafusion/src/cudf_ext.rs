use crate::task_context::CuDFTaskContext;
use datafusion::execution::TaskContext;
use std::sync::Arc;

/// Helpers for cloning a [`TaskContext`] with cuDF runtime state attached.
pub trait CuDFExt {
    fn with_cudf_task_context(&self) -> TaskContext;
}

impl CuDFExt for TaskContext {
    fn with_cudf_task_context(&self) -> TaskContext {
        task_ctx_with_extension(self, CuDFTaskContext::default())
    }
}

fn task_ctx_with_extension<T: Send + Sync + 'static>(ctx: &TaskContext, ext: T) -> TaskContext {
    TaskContext::new(
        ctx.task_id(),
        ctx.session_id(),
        ctx.session_config().clone().with_extension(Arc::new(ext)),
        ctx.scalar_functions().clone(),
        ctx.aggregate_functions().clone(),
        ctx.window_functions().clone(),
        ctx.runtime_env(),
    )
}

#[cfg(test)]
mod tests {
    use super::CuDFExt;
    use crate::task_context::CuDFTaskContext;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::execution::TaskContext;
    use datafusion::prelude::SessionConfig;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_with_cudf_task_context_installs_extension() -> Result<(), Box<dyn std::error::Error>> {
        let base = make_task_context(SessionConfig::new());
        let task_ctx = base.with_cudf_task_context();
        let ext = task_ctx.session_config().get_extension::<CuDFTaskContext>();
        assert!(ext.is_some());
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
