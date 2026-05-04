use crate::stream_source::CuDFStreamSource;
use datafusion::common::extensions_options;
use datafusion::common::{plan_err, DataFusionError};
use datafusion::config::{ConfigExtension, ConfigOptions};

extensions_options! {
    pub struct CuDFConfig {
        /// Enables CuDF optimizations.
        pub enable: bool, default = false
        /// Batch size for moving data from CPU to GPU and vice-versa.
        pub batch_size: usize, default = 8192 * 10
        /// Enables the use of multiple CUDA streams to execute GPU segments in parallel.
        pub cuda_streams: bool, default = false
        /// Allocates CUDA streams for query execution. This is private runtime wiring, not a
        /// user-facing config surface.
        pub(crate) __private_stream_source: CuDFStreamSource, default = CuDFStreamSource::default()
    }
}

impl ConfigExtension for CuDFConfig {
    const PREFIX: &'static str = "cudf";
}

impl CuDFConfig {
    /// Gets the [`CuDFConfig`] from [`ConfigOptions`]' extensions.
    pub fn from_config_options(cfg: &ConfigOptions) -> Result<&Self, DataFusionError> {
        let Some(cudf_cfg) = cfg.extensions.get::<CuDFConfig>() else {
            return plan_err!("CuDFConfig is not in ConfigOptions.extensions");
        };
        Ok(cudf_cfg)
    }

    /// Gets the [`CuDFConfig`] from [`ConfigOptions`]' extensions.
    pub fn from_config_options_mut(cfg: &mut ConfigOptions) -> Result<&mut Self, DataFusionError> {
        let Some(cudf_cfg) = cfg.extensions.get_mut::<CuDFConfig>() else {
            return plan_err!("CuDFConfig is not in ConfigOptions.extensions");
        };
        Ok(cudf_cfg)
    }

    #[allow(dead_code)]
    pub(crate) fn stream_source(&self) -> &CuDFStreamSource {
        &self.__private_stream_source
    }
}

#[cfg(test)]
mod tests {
    use super::CuDFConfig;
    use datafusion::prelude::SessionConfig;
    use std::sync::Arc;

    #[test]
    fn test_stream_source_is_available_from_session_config(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = SessionConfig::new().with_option_extension(CuDFConfig::default());
        let cudf_cfg = CuDFConfig::from_config_options(config.options())?;
        let s0 = cudf_cfg.stream_source().allocate();
        let s1 = cudf_cfg.stream_source().allocate();
        assert!(!Arc::ptr_eq(&s0, &s1));
        Ok(())
    }
}
