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
use arrow::alloc::Allocation;
use arrow::array::{make_array, ArrayData, ArrayDataBuilder, RecordBatch};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};
use cxx::UniquePtr;
use libcudf_sys::ffi::{
    cuda_default_stream_synchronize, cuda_event_create, pinned_host_alloc, pinned_host_free,
    CudaEvent, PinnedHostAlloc,
};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use crate::errors::Result;

/// Internal owner for a single pinned allocation via `cudaMallocHost`. The
/// allocation is freed automatically on drop.
struct PinnedAllocOwner {
    inner: Option<UniquePtr<PinnedHostAlloc>>,
}

// SAFETY: A pinned host allocation is plain memory addressable by both the
// host and the device. There is no thread-affinity on the CUDA side, so the
// owner can be moved across threads.
unsafe impl Send for PinnedAllocOwner {}
unsafe impl Sync for PinnedAllocOwner {}

impl PinnedAllocOwner {
    fn new(bytes: usize) -> Result<Self> {
        Ok(Self {
            inner: Some(pinned_host_alloc(bytes)?),
        })
    }

    fn capacity(&self) -> usize {
        self.inner_ref().len()
    }

    fn data_ptr(&self) -> *mut u8 {
        self.inner_ref().data() as *mut u8
    }

    fn inner_ref(&self) -> &PinnedHostAlloc {
        self.inner
            .as_ref()
            .and_then(|i| i.as_ref())
            .expect("PinnedHostAlloc should not be null")
    }
}

impl Drop for PinnedAllocOwner {
    fn drop(&mut self) {
        let Some(alloc) = self.inner.take() else {
            return;
        };
        if let Err(err) = pinned_host_free(alloc) {
            if std::thread::panicking() {
                // Already unwinding — surface the failure but don't abort by
                // double-panicking.
                eprintln!("libcudf_rs: cudaFreeHost failed during unwinding: {err}");
            } else {
                panic!("cudaFreeHost failed: {err}");
            }
        }
    }
}

/// Wrapper for [`PinnedAllocOwner`] used to pool / re-use allocations.
pub struct PinnedHostBuffer {
    inner: Option<PinnedAllocOwner>,
    requested_bytes: usize,
}

/// Process-global pool of pinned host allocations available for reuse.
///
/// `cudaMallocHost` / `cudaFreeHost` each take hundreds of microseconds, so
/// allocations are recycled here instead of being freed on drop. On a
/// `new(bytes)` request we linearly pick the smallest pooled allocation with
/// capacity >= `bytes`; the pool stays small enough that the linear scan is
/// fine. `cudaFreeHost` only runs when the pool itself drops at process
/// exit (see [`PinnedAllocOwner::drop`]).
///
/// # Why global instead of thread-local
///
/// `PinnedHostBuffer::Drop` may run on a CUDA-managed thread (the host
/// callback launched via `cudaLaunchHostFunc` to defer release until the
/// async H2D completes — see [`launch_pinned_release_on_default_stream`]).
/// A thread-local pool would route those releases into the CUDA thread's
/// pool, where worker threads can never see them, eliminating reuse and
/// causing a `cudaMallocHost` storm. A global pool ensures the buffer is
/// available to whichever thread next calls [`PinnedHostBuffer::new`].
///
/// Mutex contention is small in practice: each batch's pin/unpin path
/// touches the lock briefly (push/pop on a Vec) and per-iter allocation
/// counts are in the thousands across many threads.
static PINNED_POOL: OnceLock<Mutex<Vec<PinnedAllocOwner>>> = OnceLock::new();

fn pinned_pool() -> &'static Mutex<Vec<PinnedAllocOwner>> {
    PINNED_POOL.get_or_init(|| Mutex::new(Vec::new()))
}

#[cfg(test)]
fn pool_len() -> usize {
    pinned_pool().lock().unwrap().len()
}

/// Drop every cached allocation. Drains via [`PinnedAllocOwner::drop`], so
/// any `cudaFreeHost` failure becomes a panic here. Test-only — production
/// code never needs to drain explicitly.
#[cfg(test)]
fn drain_pool() {
    pinned_pool().lock().unwrap().clear();
}

impl PinnedHostBuffer {
    /// Allocate `bytes` of pinned host memory, reusing a pooled buffer if
    /// one of sufficient capacity is available in the global pool.
    pub fn new(bytes: usize) -> Result<Self> {
        let pooled = {
            let mut pool = pinned_pool().lock().expect("PINNED_POOL poisoned");
            // Pick the smallest pooled buffer with capacity >= requested.
            let pos = pool
                .iter()
                .enumerate()
                .filter(|(_, owner)| owner.capacity() >= bytes)
                .min_by_key(|(_, owner)| owner.capacity())
                .map(|(i, _)| i);
            pos.map(|i| pool.swap_remove(i))
        };
        let inner = match pooled {
            Some(owner) => owner,
            None => PinnedAllocOwner::new(bytes)?,
        };
        Ok(Self {
            inner: Some(inner),
            requested_bytes: bytes,
        })
    }

    /// Number of bytes the caller requested. May be less than the underlying
    /// allocation's capacity if it came from the pool.
    pub fn len(&self) -> usize {
        self.requested_bytes
    }

    /// Whether the requested allocation is zero-sized.
    pub fn is_empty(&self) -> bool {
        self.requested_bytes == 0
    }

    /// Raw pointer to the start of the allocation.
    pub fn as_ptr(&self) -> *mut u8 {
        self.inner
            .as_ref()
            .expect("PinnedHostBuffer must own an allocation")
            .data_ptr()
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if let Some(owner) = self.inner.take() {
            // Return to the global pool for reuse rather than freeing.
            // The actual `cudaFreeHost` happens at process exit when the
            // OnceLock drops the inner Mutex, via `PinnedAllocOwner::drop`.
            //
            // This Drop may run on either a worker thread (when no GPU
            // ownership transfer happened) or on a CUDA-managed callback
            // thread (when the table that owned this buffer scheduled a
            // stream-ordered release). Either way the global pool serves
            // both producers.
            if let Ok(mut pool) = pinned_pool().lock() {
                pool.push(owner);
            } else {
                // Pool mutex poisoned — let `owner` drop here, which calls
                // `cudaFreeHost` synchronously. Slower but safe.
            }
        }
    }
}

/// Block until all GPU work submitted to the CUDA default stream has
/// completed.
///
/// Required after issuing an asynchronous upload from a pinned source if the
/// source is about to be dropped, since `cudaMemcpyAsync` returns before the
/// DMA has finished and the pinned buffer must outlive the transfer.
pub fn synchronize_default_stream() -> Result<()> {
    cuda_default_stream_synchronize()?;
    Ok(())
}

/// A [`RecordBatch`] whose host buffers are all pinned, plus the
/// `Arc<PinnedHostBuffer>` keepalives that anchor those buffers' lifetimes.
///
/// Returned by [`pin_record_batch`]. Callers that only need the
/// [`RecordBatch`] can drop `buffers`; callers that need to defer the
/// pinned source's release until a stream-ordered point should keep them.
pub struct PinnedBatch {
    pub batch: RecordBatch,
    pub buffers: Vec<Arc<PinnedHostBuffer>>,
}

/// Return a [`PinnedBatch`] whose underlying buffers all live in pinned
/// (page-locked) host memory, plus a vector of `Arc<PinnedHostBuffer>`
/// keepalives — one per leaf buffer that was pinned.
///
/// The schema, lengths, offsets, and null counts of every column are
/// preserved exactly; only the host-side storage of each leaf
/// [`arrow::buffer::Buffer`] is replaced with a pinned-backed copy.
///
/// Empty buffers are passed through unchanged because `cudaMallocHost(0)` is
/// not portable and a zero-byte buffer has no data to DMA.
///
/// The caller must keep `buffers` alive until any async H2D copy reading
/// from `batch`'s storage has completed. The intended use is to attach
/// `buffers` to the resulting `CuDFTable` via
/// [`crate::CuDFTable::with_pinned_keepalive`], which schedules a
/// stream-ordered release once the H2D drains.
pub fn pin_record_batch(batch: RecordBatch) -> Result<PinnedBatch> {
    let schema = batch.schema();
    let mut buffers = Vec::new();
    let arrays = batch
        .columns()
        .iter()
        .map(|arr| pin_array_data(arr.to_data(), &mut buffers).map(make_array))
        .collect::<Result<Vec<_>>>()?;
    Ok(PinnedBatch {
        batch: RecordBatch::try_new(schema, arrays)?,
        buffers,
    })
}

fn pin_array_data(
    data: ArrayData,
    out_buffers: &mut Vec<Arc<PinnedHostBuffer>>,
) -> Result<ArrayData> {
    let buffers = data
        .buffers()
        .iter()
        .map(|b| pin_buffer(b, out_buffers))
        .collect::<Result<Vec<_>>>()?;

    let children = data
        .child_data()
        .iter()
        .cloned()
        .map(|c| pin_array_data(c, out_buffers))
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
        builder = builder.nulls(Some(pin_null_buffer(nulls, out_buffers)?));
    }

    // SAFETY: only the storage of each leaf buffer is replaced; data type,
    // lengths, offsets, and null counts are preserved. The new ArrayData is
    // structurally identical to the input.
    Ok(unsafe { builder.build_unchecked() })
}

fn pin_null_buffer(
    nulls: &NullBuffer,
    out_buffers: &mut Vec<Arc<PinnedHostBuffer>>,
) -> Result<NullBuffer> {
    let bool_buf = nulls.inner();
    let pinned = pin_buffer(bool_buf.inner(), out_buffers)?;
    let new_bool = BooleanBuffer::new(pinned, bool_buf.offset(), bool_buf.len());
    // SAFETY: `pin_buffer` copies the underlying bytes verbatim, so the bit
    // pattern (and therefore the null count) is preserved.
    Ok(unsafe { NullBuffer::new_unchecked(new_bool, nulls.null_count()) })
}

fn pin_buffer(buf: &Buffer, out_buffers: &mut Vec<Arc<PinnedHostBuffer>>) -> Result<Buffer> {
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
    }
    out_buffers.push(Arc::clone(&pinned));
    let arrow_buf = unsafe {
        Buffer::from_custom_allocation(
            NonNull::new(dst).expect("pinned allocation pointer is non-null"),
            bytes,
            pinned as Arc<dyn Allocation>,
        )
    };
    Ok(arrow_buf)
}

/// Owning Rust wrapper for a CUDA event with `cudaEventDisableTiming`.
///
/// Used by [`PinRing`] as a per-slot fence: the ring records this event on
/// the CUDA default stream after the H2D copy that consumed the slot's
/// pinned buffers, and synchronizes on it before reusing the slot. This
/// gives stream-ordered backpressure without a host-blocking
/// `cudaStreamSynchronize` per batch.
pub struct CuDFEvent {
    inner: UniquePtr<CudaEvent>,
}

// SAFETY: `cudaEvent_t` is a process-global opaque handle. The CUDA runtime
// allows recording, querying, synchronizing, and destroying events from
// any host thread. Moving the wrapper across threads is therefore safe,
// and concurrent `&self` calls into `record_on_default_stream` / `query` /
// `synchronize` are also safe (the underlying APIs are thread-safe).
unsafe impl Send for CuDFEvent {}
unsafe impl Sync for CuDFEvent {}

impl CuDFEvent {
    /// Create a fresh event with `cudaEventDisableTiming`.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: cuda_event_create()?,
        })
    }

    /// Record this event at the current point in the CUDA default stream.
    pub fn record_on_default_stream(&self) -> Result<()> {
        self.inner_ref().record_on_default_stream()?;
        Ok(())
    }

    /// Non-blocking check: returns `true` if the event has fired.
    pub fn query(&self) -> Result<bool> {
        Ok(self.inner_ref().query()?)
    }

    /// Block the calling thread until the event fires.
    pub fn synchronize(&self) -> Result<()> {
        self.inner_ref().synchronize()?;
        Ok(())
    }

    fn inner_ref(&self) -> &CudaEvent {
        self.inner.as_ref().expect("CudaEvent should not be null")
    }
}

/// One slot in [`PinRing`]. Owns a set of pinned host buffers that back the
/// most recent batch put through this slot, plus the event recorded after
/// that batch's H2D. The slot's buffers stay alive (held by the slot's
/// `Arc`s) until the next time this slot is recycled — at which point the
/// ring synchronizes on `event` first, guaranteeing the H2D has retired
/// before the buffers can be safely overwritten.
struct PinSlot {
    buffers: Vec<Arc<PinnedHostBuffer>>,
    event: CuDFEvent,
    in_use: bool,
}

/// Bounded-pipelining pool of pinned host buffers driven by per-slot CUDA
/// events.
///
/// `PinRing` lets a per-batch upload pipeline stay at most `kRing` batches
/// ahead of the GPU without ever calling `cudaStreamSynchronize` on the
/// host. Each batch claims the next slot in round-robin order; if that
/// slot is on its second-or-later turn, the ring blocks on its event
/// before allowing reuse. In steady state the event has long since fired,
/// the wait is a no-op, and the host runs uncontested.
///
/// Sizing notes (see `docs/` for the analysis):
/// - `kRing = 3` covers CPU-fill / H2D / kernel as three concurrent stages
///   and is the recommended default.
/// - `kRing = 2` saves ~33% pinned memory at the cost of less overlap
///   slack.
/// - Memory cost is `kRing × per_batch_pinned_bytes`, typically megabytes.
///
/// The ring is **not** thread-safe by itself — it is intended to live
/// inside one `CuDFLoadExec`'s per-batch processing loop, where exactly
/// one task pins/uploads at a time.
pub struct PinRing {
    slots: Vec<PinSlot>,
    cursor: usize,
}

impl PinRing {
    /// Create a new ring with `k_ring` empty slots. Each slot's pinned
    /// buffers are allocated lazily on first use.
    pub fn new(k_ring: usize) -> Result<Self> {
        assert!(k_ring >= 2, "kRing must be at least 2 for pipelining");
        let slots = (0..k_ring)
            .map(|_| {
                Ok(PinSlot {
                    buffers: Vec::new(),
                    event: CuDFEvent::new()?,
                    in_use: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { slots, cursor: 0 })
    }

    /// Number of slots in the ring.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the ring has zero slots (impossible per `new`'s assert,
    /// but the method exists to satisfy clippy).
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Claim the next slot, copy `batch`'s buffers into freshly-pinned
    /// host memory associated with that slot, and return a [`RecordBatch`]
    /// backed by those pinned buffers.
    ///
    /// If the slot was used at least once before, this synchronizes on
    /// the slot's event first — the only host-blocking operation in the
    /// ring's hot path, and only when the host is racing ahead of the
    /// GPU. After this call returns, the caller must consume the
    /// returned `RecordBatch` (typically via `CuDFTable::from_arrow_host`)
    /// and then call [`Self::record_h2d_event`] before the next
    /// `fill_next`.
    pub fn fill_next(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        let slot = &mut self.slots[self.cursor];
        if slot.in_use {
            // Backpressure: only blocks if the GPU hasn't yet drained past
            // this slot's last H2D. In steady state this is a no-op.
            slot.event.synchronize()?;
        }
        // The previous batch's pinned `Arc`s drop here, going back to the
        // global pool. The slot's event guarantees the corresponding H2D
        // has retired, so the OS pages are safe to re-issue.
        slot.buffers.clear();
        let schema = batch.schema();
        let arrays = batch
            .columns()
            .iter()
            .map(|arr| pin_array_data(arr.to_data(), &mut slot.buffers).map(make_array))
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new(schema, arrays)?)
    }

    /// Record the just-filled slot's event on the CUDA default stream and
    /// advance the ring cursor. Must be called after the H2D for the
    /// `RecordBatch` returned by [`Self::fill_next`] has been queued
    /// (e.g. inside `cuDF::from_arrow_host`).
    pub fn record_h2d_event(&mut self) -> Result<()> {
        let slot = &mut self.slots[self.cursor];
        slot.event.record_on_default_stream()?;
        slot.in_use = true;
        self.cursor = (self.cursor + 1) % self.slots.len();
        Ok(())
    }
}

impl Drop for PinRing {
    fn drop(&mut self) {
        // Make sure any in-flight H2Ds reading from slot buffers have
        // retired before the slot `Arc`s drop. Without this, dropping the
        // ring while the GPU is still using a slot's source memory would
        // race with the buffers' return to the global pool.
        for slot in &self.slots {
            if slot.in_use {
                let _ = slot.event.synchronize();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn pinned_host_buffer_round_trip() -> Result<()> {
        let buf = PinnedHostBuffer::new(64)?;
        assert_eq!(buf.len(), 64);
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
            .batch
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
            .batch
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

    /// Dropping a [`PinnedHostBuffer`] should return its allocation to the
    /// thread-local pool so the next allocation of the same size reuses it
    /// instead of calling `cudaMallocHost` again.
    #[test]
    fn drop_returns_buffer_to_pool() -> Result<()> {
        drain_pool();

        let buf = PinnedHostBuffer::new(2048)?;
        let ptr_before = buf.as_ptr() as usize;
        assert_eq!(pool_len(), 0);
        drop(buf);
        assert_eq!(
            pool_len(),
            1,
            "drop should push the allocation into the pool"
        );

        let buf2 = PinnedHostBuffer::new(2048)?;
        assert_eq!(
            buf2.as_ptr() as usize,
            ptr_before,
            "pool should return the same allocation"
        );
        assert_eq!(
            pool_len(),
            0,
            "reuse should remove the allocation from the pool"
        );
        Ok(())
    }

    /// When a request can be served by multiple pooled allocations, the pool
    /// should return the smallest one whose capacity is at least the request,
    /// to avoid permanently inflating the working set.
    #[test]
    fn pool_picks_smallest_sufficient() -> Result<()> {
        drain_pool();

        let small = PinnedHostBuffer::new(1024)?;
        let medium = PinnedHostBuffer::new(4096)?;
        let large = PinnedHostBuffer::new(16_384)?;
        let small_ptr = small.as_ptr() as usize;
        let medium_ptr = medium.as_ptr() as usize;
        let large_ptr = large.as_ptr() as usize;
        drop(small);
        drop(medium);
        drop(large);

        // 3 KiB request — the 4 KiB allocation is the smallest sufficient.
        let pick = PinnedHostBuffer::new(3000)?;
        let pick_ptr = pick.as_ptr() as usize;
        assert_eq!(pick_ptr, medium_ptr, "expected the 4 KiB pooled allocation");
        assert_ne!(pick_ptr, small_ptr);
        assert_ne!(pick_ptr, large_ptr);
        Ok(())
    }

    /// `PinnedHostBuffer::Drop` must be safe to run during unwinding — it
    /// returns to the pool without panicking, so a user `panic!` while a
    /// pinned batch is in flight should be caught cleanly.
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
}
