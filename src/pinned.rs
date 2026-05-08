//! Pinned host memory utilities for accelerating Arrow → GPU uploads.
//!
//! When the host source of a `cudaMemcpyAsync` is pageable memory (e.g. a Rust
//! `Vec` or an arrow-rs default-allocator buffer), the CUDA driver must first
//! stage the data through a single device-wide pinned staging buffer before
//! kicking off the DMA. That staging step is synchronous. All `cudaMemcpyAsync`
//! calls on a stream will serialize.
//!
//! When the source is page-locked ("pinned") memory allocated via
//! `cudaMallocHost`, the driver can DMA directly from the source and the call
//! is fully asynchronous.
use crate::config::ensure_pools_configured;
use crate::errors::{CuDFError, Result};
use arrow::alloc::Allocation;
use arrow::array::{make_array, ArrayData, ArrayDataBuilder, RecordBatch};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow::error::ArrowError;
use cxx::UniquePtr;
use libcudf_sys::ffi::{
    cuda_default_stream_synchronize, cuda_event_create_with_flags, get_default_stream,
    get_pinned_memory_resource, CudaEvent, HostDeviceAsyncResourceRef,
};
use std::collections::VecDeque;
use std::mem;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;
use tokio::sync::mpsc::{
    self,
    error::{SendError, TryRecvError},
    UnboundedReceiver, UnboundedSender,
};

const PINNED_REAPER_ENV: &str = "LIBCUDF_PINNED_UPLOAD_REAPER";
const GROUP_INTERVAL: Duration = Duration::from_millis(5);
const POLL_INTERVAL: Duration = Duration::from_millis(1);
const MAX_GROUP_BATCHES: usize = 32;
const CUDA_EVENT_DISABLE_TIMING: u32 = 2;

/// Process-global handle to cuDF's pinned MR. Lazily initialized once;
/// thereafter every alloc/dealloc reads through the same `&'static` handle.
/// The underlying cuDF MR (a `pinned_pool_with_fallback_memory_resource`
/// backed by `rmm::mr::pool_memory_resource`) is itself thread-safe.
fn pinned_mr() -> &'static HostDeviceAsyncResourceRef {
    static MR: OnceLock<UniquePtr<HostDeviceAsyncResourceRef>> = OnceLock::new();
    MR.get_or_init(get_pinned_memory_resource)
        .as_ref()
        .expect("pinned MR is null")
}

/// RAII wrapper for a single pinned host allocation drawn from cuDF's pinned
/// memory resource. The pool is process-global and shared with cuDF's own
/// pinned-memory consumers (e.g. the download path).
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: A pinned host allocation is plain memory addressable by both the
// host and the device. There is no thread-affinity on the CUDA side, so the
// buffer can be moved across threads.
unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    /// Allocate `bytes` of pinned host memory from cuDF's pinned MR.
    fn new(bytes: usize) -> Result<Self> {
        let ptr = pinned_mr().allocate_sync(bytes)? as *mut u8;
        Ok(Self { ptr, bytes })
    }

    /// Raw pointer to the start of the allocation.
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            pinned_mr().deallocate_sync(self.ptr as usize, self.bytes);
        }
    }
}

/// Block until all GPU work submitted to cuDF's default stream has
/// completed.
///
/// Required after issuing an asynchronous upload from a pinned source if the
/// source is about to be dropped, since `cudaMemcpyAsync` returns before the
/// DMA has finished and the pinned buffer must outlive the transfer.
pub fn synchronize_default_stream() -> Result<()> {
    cuda_default_stream_synchronize()?;
    Ok(())
}

/// Return whether pinned upload batches should be retained by the async reaper.
pub fn pinned_upload_reaper_enabled() -> bool {
    std::env::var(PINNED_REAPER_ENV).map_or(true, |v| v != "0")
}

/// Submit a pinned batch whose cuDF upload has already been enqueued.
///
/// The reaper keeps the batch's pinned host buffers alive until a CUDA event
/// recorded after the upload completes.
pub fn submit_uploaded_pinned_batch(batch: RecordBatch) -> Result<()> {
    let reaper = match pinned_upload_reaper() {
        Ok(reaper) => reaper,
        Err(err) => {
            synchronize_default_stream()?;
            drop(batch);
            return Err(err);
        }
    };

    reaper.submit(batch)
}

fn pinned_upload_reaper() -> Result<&'static PinnedUploadReaper> {
    static REAPER: OnceLock<PinnedUploadReaper> = OnceLock::new();

    if let Some(reaper) = REAPER.get() {
        return Ok(reaper);
    }

    let reaper = PinnedUploadReaper::try_spawn()?;
    let _ = REAPER.set(reaper);
    Ok(REAPER
        .get()
        .expect("pinned upload reaper should be initialized"))
}

struct PinnedUploadReaper {
    tx: UnboundedSender<RecordBatch>,
}

impl PinnedUploadReaper {
    fn try_spawn() -> Result<Self> {
        let handle = Handle::try_current().map_err(|_| {
            CuDFError::ArrowError(ArrowError::InvalidArgumentError(format!(
                "{PINNED_REAPER_ENV}=1 requires an active Tokio runtime"
            )))
        })?;
        let (tx, rx) = mpsc::unbounded_channel();
        drop(handle.spawn(run_pinned_upload_reaper(rx)));
        Ok(Self { tx })
    }

    fn submit(&self, batch: RecordBatch) -> Result<()> {
        self.tx.send(batch).map_err(|err| {
            synchronize_before_dropping_send_error(err);
            CuDFError::ArrowError(ArrowError::InvalidArgumentError(
                "pinned upload reaper task is not running".to_string(),
            ))
        })
    }
}

struct SealedGroup {
    batches: Vec<RecordBatch>,
    event: UniquePtr<CudaEvent>,
}

async fn run_pinned_upload_reaper(mut rx: UnboundedReceiver<RecordBatch>) {
    let mut open = Vec::<RecordBatch>::new();
    let mut opened_at = Instant::now();
    let mut sealed = VecDeque::<SealedGroup>::new();
    let mut disconnected = false;

    while !disconnected {
        match tokio::time::timeout(POLL_INTERVAL, rx.recv()).await {
            Ok(Some(batch)) => {
                push_open_batch(&mut open, &mut opened_at, batch);
                while open.len() < MAX_GROUP_BATCHES {
                    match rx.try_recv() {
                        Ok(batch) => push_open_batch(&mut open, &mut opened_at, batch),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
            }
            Ok(None) => disconnected = true,
            Err(_) => {}
        }

        if open.len() >= MAX_GROUP_BATCHES {
            seal_open_group_or_panic(&mut open, &mut opened_at, &mut sealed);
        }

        if !open.is_empty() && opened_at.elapsed() >= GROUP_INTERVAL {
            seal_open_group_or_panic(&mut open, &mut opened_at, &mut sealed);
        }

        reap_completed_groups_or_panic(&mut sealed);
    }

    if !open.is_empty() {
        seal_open_group_or_panic(&mut open, &mut opened_at, &mut sealed);
    }

    for group in sealed {
        group.event.synchronize().unwrap_or_else(|err| {
            panic!("synchronize upload event: {err}");
        });
        drop(group.batches);
    }
}

fn push_open_batch(open: &mut Vec<RecordBatch>, opened_at: &mut Instant, batch: RecordBatch) {
    if open.is_empty() {
        *opened_at = Instant::now();
    }
    open.push(batch);
}

fn seal_open_group(
    open: &mut Vec<RecordBatch>,
    opened_at: &mut Instant,
    sealed: &mut VecDeque<SealedGroup>,
) -> Result<()> {
    if open.is_empty() {
        *opened_at = Instant::now();
        return Ok(());
    }

    let stream = get_default_stream();
    let event = cuda_event_create_with_flags(CUDA_EVENT_DISABLE_TIMING)?;
    event.record(stream.as_ref().expect("default stream should not be null"))?;

    sealed.push_back(SealedGroup {
        batches: mem::take(open),
        event,
    });
    *opened_at = Instant::now();
    Ok(())
}

fn reap_completed_groups(sealed: &mut VecDeque<SealedGroup>) -> Result<usize> {
    let mut completed = 0;
    while let Some(group) = sealed.front() {
        if !group.event.query()? {
            break;
        }
        let _ = sealed.pop_front();
        completed += 1;
    }
    Ok(completed)
}

fn seal_open_group_or_panic(
    open: &mut Vec<RecordBatch>,
    opened_at: &mut Instant,
    sealed: &mut VecDeque<SealedGroup>,
) {
    seal_open_group(open, opened_at, sealed).unwrap_or_else(|err| {
        let _ = synchronize_default_stream();
        panic!("record upload event: {err}");
    });
}

fn reap_completed_groups_or_panic(sealed: &mut VecDeque<SealedGroup>) {
    reap_completed_groups(sealed).unwrap_or_else(|err| {
        let _ = synchronize_default_stream();
        panic!("query upload events: {err}");
    });
}

fn synchronize_before_dropping_send_error(err: SendError<RecordBatch>) {
    let SendError(batch) = err;
    let _ = synchronize_default_stream();
    drop(batch);
}

/// Return a copy of `batch` whose underlying buffers all live in pinned
/// (page-locked) host memory.
///
/// The schema, lengths, offsets, and null counts of every column are
/// preserved exactly; only the host-side storage of each leaf
/// [`Buffer`] is replaced with a pinned-backed copy.
///
/// Empty buffers are passed through unchanged because `cudaMallocHost(0)` is
/// not portable and a zero-byte buffer has no data to DMA.
pub fn pin_record_batch(batch: RecordBatch) -> Result<RecordBatch> {
    ensure_pools_configured();
    let schema = batch.schema();
    let arrays = batch
        .columns()
        .iter()
        .map(|arr| pin_array_data(arr.to_data()).map(make_array))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(schema, arrays)?)
}

fn pin_array_data(data: ArrayData) -> Result<ArrayData> {
    let buffers = data
        .buffers()
        .iter()
        .map(pin_buffer)
        .collect::<Result<Vec<_>>>()?;

    let children = data
        .child_data()
        .iter()
        .cloned()
        .map(pin_array_data)
        .collect::<Result<Vec<_>>>()?;

    let mut builder = ArrayDataBuilder::new(data.data_type().clone())
        .len(data.len())
        .offset(data.offset())
        .buffers(buffers)
        .child_data(children);

    // The null mask is small (1 bit per row), but leaving it pageable means
    // every nullable column still pays the per-call staging cost (~30-60 µs)
    // on its `cudaMemcpyAsync`. Pinning it makes the upload uniformly async.
    if let Some(nulls) = data.nulls() {
        builder = builder.nulls(Some(pin_null_buffer(nulls)?));
    }

    // SAFETY: only the storage of each leaf buffer is replaced; data type,
    // lengths, offsets, and null counts are preserved. The new ArrayData is
    // structurally identical to the input.
    Ok(unsafe { builder.build_unchecked() })
}

fn pin_null_buffer(nulls: &NullBuffer) -> Result<NullBuffer> {
    let bool_buf = nulls.inner();
    let pinned = pin_buffer(bool_buf.inner())?;
    let new_bool = BooleanBuffer::new(pinned, bool_buf.offset(), bool_buf.len());
    // SAFETY: `pin_buffer` copies the underlying bytes verbatim, so the bit
    // pattern (and therefore the null count) is preserved.
    Ok(unsafe { NullBuffer::new_unchecked(new_bool, nulls.null_count()) })
}

fn pin_buffer(buf: &Buffer) -> Result<Buffer> {
    let bytes = buf.len();
    if bytes == 0 {
        return Ok(buf.clone());
    }

    let pinned = Arc::new(PinnedHostBuffer::new(bytes)?);
    let dst = pinned.as_ptr();
    // SAFETY: `pinned` was just allocated with at least `bytes` capacity,
    // `buf.as_ptr()` is valid for `bytes` reads, and the regions do not
    // overlap (different allocations).
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, bytes);
        Ok(Buffer::from_custom_allocation(
            NonNull::new(dst).expect("pinned allocation pointer is non-null"),
            bytes,
            pinned as Arc<dyn Allocation>,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::error::ArrowError;
    use std::sync::Arc;

    #[test]
    fn pinned_host_buffer_round_trip() -> Result<()> {
        let buf = PinnedHostBuffer::new(64)?;
        // SAFETY: we own the allocation and it has 64 bytes of capacity.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(buf.as_ptr(), 64);
            slice.fill(0xAB);
            assert!(slice.iter().all(|b| *b == 0xAB));
        }
        Ok(())
    }

    #[test]
    fn pin_record_batch_preserves_primitive_data() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let values: Vec<i64> = (0..1024).collect();
        let arr = Int64Array::from(values.clone());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])?;

        let pinned = pin_record_batch(batch)?;
        let out = pinned
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        assert_eq!(out.len(), values.len());
        for (i, expected) in values.iter().enumerate() {
            assert_eq!(out.value(i), *expected);
        }
        Ok(())
    }

    #[test]
    fn pin_record_batch_preserves_variable_width_data() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let arr = StringArray::from(vec![Some("alpha"), None, Some("beta"), Some("gamma")]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])?;

        let pinned = pin_record_batch(batch)?;
        let out = pinned
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        assert_eq!(out.len(), 4);
        assert_eq!(out.value(0), "alpha");
        assert!(out.is_null(1));
        assert_eq!(out.value(2), "beta");
        assert_eq!(out.value(3), "gamma");
        Ok(())
    }

    #[test]
    fn submit_uploaded_pinned_batch_round_trips_nullable_batch(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;

        rt.block_on(async {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
            let arr = Int64Array::from(vec![Some(11), None, Some(13), Some(17)]);
            let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])?;
            let pinned = pin_record_batch(batch)?;

            let table = crate::CuDFTable::from_arrow_host(pinned.clone())?;
            submit_uploaded_pinned_batch(pinned)?;
            tokio::time::sleep(Duration::from_millis(20)).await;

            let output = table.into_view().to_arrow_host()?;
            let out = output
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64Array");
            assert_eq!(out.len(), 4);
            assert_eq!(out.value(0), 11);
            assert!(out.is_null(1));
            assert_eq!(out.value(2), 13);
            assert_eq!(out.value(3), 17);

            Ok::<(), Box<dyn std::error::Error>>(())
        })?;

        Ok(())
    }

    #[test]
    fn push_open_batch_starts_group_interval_on_first_batch(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut open = Vec::new();
        let mut opened_at = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .expect("subtracting a minute from now should be valid");

        push_open_batch(&mut open, &mut opened_at, empty_i64_batch()?);
        assert_eq!(open.len(), 1);
        assert!(opened_at.elapsed() < Duration::from_secs(1));

        let first_opened_at = opened_at;
        push_open_batch(&mut open, &mut opened_at, empty_i64_batch()?);
        assert_eq!(open.len(), 2);
        assert_eq!(opened_at, first_opened_at);

        Ok(())
    }

    /// `PinnedHostBuffer::Drop` must be safe to run during unwinding — a
    /// user `panic!` while a pinned batch is in flight should propagate
    /// cleanly without double-panic.
    #[test]
    fn drop_during_unwinding_does_not_double_panic() {
        let result = std::panic::catch_unwind(|| {
            let _buf = PinnedHostBuffer::new(1024).expect("alloc");
            panic!("simulated user panic");
        });
        assert!(
            result.is_err(),
            "outer panic should propagate to catch_unwind"
        );
    }

    fn empty_i64_batch() -> std::result::Result<RecordBatch, ArrowError> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let arr = Int64Array::from(Vec::<i64>::new());
        RecordBatch::try_new(schema, vec![Arc::new(arr)])
    }
}
