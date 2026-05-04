//! DataFusion integration for GPU-accelerated query execution with cuDF
//!
//! This crate provides custom ExecutionPlan nodes that execute operations
//! on the GPU using NVIDIA's cuDF library through DataFusion.

pub mod aggregate;
mod cudf_ext;
mod errors;
mod expr;
mod optimizer;
mod physical;
mod stream_source;
mod task_context;

#[cfg(any(feature = "integration", test))]
pub mod test_utils;

pub use cudf_ext::CuDFExt;
pub use libcudf_rs::configure_default_pools;
pub use libcudf_rs::DevicePoolConfig;
pub use libcudf_rs::PinnedPoolConfig;
pub use optimizer::{CuDFConfig, HostToCuDFRule};
pub use physical::{CuDFLoadExec, CuDFUnloadExec};
pub use task_context::CuDFTaskContext;
