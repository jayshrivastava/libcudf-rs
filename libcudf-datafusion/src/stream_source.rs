use datafusion::common::{not_impl_err, Result};
use datafusion::config::{ConfigField, Visit};
use libcudf_rs::{CuDFStream, CuDFStreamFlags};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

/// Allocates CUDA streams for cuDF execution.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct CuDFStreamSource;

impl CuDFStreamSource {
    #[allow(dead_code)]
    pub(crate) fn allocate(&self) -> Arc<CuDFStream> {
        Arc::new(CuDFStream::with_flags(CuDFStreamFlags::NonBlocking))
    }
}

impl ConfigField for CuDFStreamSource {
    fn visit<V: Visit>(&self, _: &mut V, _: &str, _: &'static str) {}

    fn set(&mut self, _: &str, _: &str) -> Result<()> {
        not_impl_err!("Not implemented")
    }
}

impl Debug for CuDFStreamSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CuDFStreamSource")
    }
}

#[cfg(test)]
mod tests {
    use super::CuDFStreamSource;
    use std::sync::Arc;

    #[test]
    fn test_allocate_returns_distinct_streams() -> Result<(), Box<dyn std::error::Error>> {
        let source = CuDFStreamSource;
        let s0 = source.allocate();
        let s1 = source.allocate();
        assert!(!Arc::ptr_eq(&s0, &s1));
        Ok(())
    }
}
