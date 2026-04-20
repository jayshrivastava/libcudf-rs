use cxx::UniquePtr;

/// Stream creation flags for CUDA stream-backed cuDF execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuDFStreamFlags {
    /// Create a stream that synchronizes with the default stream.
    SyncDefault,
    /// Create a non-blocking stream that does not synchronize with the default stream.
    NonBlocking,
}

const CUDA_STREAM_FLAG_SYNC_DEFAULT: u32 = 0;
const CUDA_STREAM_FLAG_NON_BLOCKING: u32 = 1;

impl From<CuDFStreamFlags> for u32 {
    fn from(value: CuDFStreamFlags) -> Self {
        match value {
            CuDFStreamFlags::SyncDefault => CUDA_STREAM_FLAG_SYNC_DEFAULT,
            CuDFStreamFlags::NonBlocking => CUDA_STREAM_FLAG_NON_BLOCKING,
        }
    }
}

/// Owning Rust wrapper for an opaque CUDA stream handle.
///
/// The actual stream lifetime is managed by the underlying C++ `rmm::cuda_stream`.
pub struct CuDFStream {
    inner: UniquePtr<libcudf_sys::ffi::CudaStream>,
}

impl CuDFStream {
    /// Create a stream using the sync-default creation flag.
    pub fn new() -> Self {
        Self {
            inner: libcudf_sys::ffi::cuda_stream_create(),
        }
    }

    /// Create a stream with explicit creation flags.
    pub fn with_flags(flags: CuDFStreamFlags) -> Self {
        Self {
            inner: libcudf_sys::ffi::cuda_stream_create_with_flags(flags.into()),
        }
    }

    /// Get a reference to the underlying FFI stream handle.
    pub(crate) fn inner(&self) -> &libcudf_sys::ffi::CudaStream {
        self.inner.as_ref().expect("CudaStream should not be null")
    }
}

impl Default for CuDFStream {
    fn default() -> Self {
        Self::new()
    }
}
