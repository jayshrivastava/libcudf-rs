AI Generated

Aggregate CUDA Streams: Why the Speedup Is Small

Cuda streams only have a small impact on the aggregate benchmark (~1.05-1.10x) 

Streams are plumbed correctly. The aggregate bench is 92% HtoD memcpy by
GPU time, and those copies serialize across streams. There are three
layered reasons; pinning the input fixes the first one but exposes the
other two.

Trace summary

nsys profile of combined/gpu_streams_on/20000000, 4 partitions:

Op

% time

Per-iter

HtoD memcpy 

92% 

~813 

DtoH memcpy 

7% 

~84 

DtoD memcpy 

1% 

~147 

Kernels 

tiny 

— 

Streams 49-52 (Partial segment) and 45-48 (Final, after CPU repartition) are
the four partition streams. Their HtoD copies run back-to-back, never
overlapped — ~106 µs each, ~170 µs apart.

Why copies serialize

                       host                            GPU
                      ┌──────────────────┐    ┌────────────────────┐
PAGEABLE source       │ Rust Vec / Arrow │    │ device buffer      │
(default arrow-rs):   │  ┌────────────┐  │    │  ┌──────────────┐  │
                      │  │ batch data │──┼────┼─▶│ column data  │  │
                      │  └─────┬──────┘  │    │  └──────────────┘  │
                      │        │ cpu     │    │                    │
                      │        ▼ memcpy  │    │                    │
                      │  ┌────────────┐  │    │                    │
                      │  │ driver     │ DMA   │                    │
                      │  │ STAGING    │──┼────┼──────▶ ...         │
                      │  │ (single,   │  │    │                    │
                      │  │  shared)   │  │    │                    │
                      │  └────────────┘  │    │                    │
                      └──────────────────┘    └────────────────────┘

  All 4 partition streams take turns at the same staging buffer.
  Critical-section ⇒ serialization. ~92% of GPU time waits here.

PINNED source         ┌──────────────────┐    ┌────────────────────┐
(cudaMallocHost):     │ pinned host buf  │ DMA│ device buffer      │
                      │  ┌────────────┐  │    │  ┌──────────────┐  │
                      │  │ batch data │──┼────┼─▶│ column data  │  │
                      │  └────────────┘  │    │  └──────────────┘  │
                      └──────────────────┘    └────────────────────┘

  No staging, direct DMA. Two streams can run on T4's 2 copy engines.
  But: cuDF/RMM sync on the host side serialize the *issuing* of work.

Timeline view

What this looks like over time on a single stream. Pageable forces the CPU
to stage every copy through a shared driver buffer before the DMA can start,
and all four partition streams contend for the same staging slot:

PAGEABLE, 1 stream (today):
  CPU   [stage col0][--wait-DMA--][stage col1][--wait-DMA--][stage col2]...
  PCIe              [---DMA col0---]          [---DMA col1---]

PAGEABLE, 4 streams (today, what nsys shows):
  CPU   [stage A][wait][stage B][wait][stage C][wait][stage D][wait]...
  PCIe          [DMA A]         [DMA B]        [DMA C]        [DMA D]
                  ─── serialized on the shared staging slot ───

With pinned source, no staging copy. The CPU just enqueues and the DMA
fires; multiple streams' DMAs can interleave on the GPU's copy engines:

PINNED, 1 stream (per-copy faster, no staging step):
  CPU   [enq][enq][enq]...
  PCIe  [-------- DMA col0 --------][-------- DMA col1 --------]...

PINNED, 4 streams (what we *want* to see):
  CPU   [enq A][enq B][enq C][enq D]...
  PCIe  [---- DMA A ----]
              [---- DMA B ----]                 ← T4 has 2 copy engines,
                    [---- DMA C ----]             so up to 2 in flight at once.
                          [---- DMA D ----]

The smoking-gun test got us pinned source but bottlenecks 2 and 3 still
serialize the CPU enqueue side, so the pinned-4-streams picture above
is not yet what we observe — we see the same pageable-style staircase.

Three layered bottlenecks

1. Pageable staging. Before any DMA, pageable host pages must be copied
into a single device-wide pinned staging slot. Critical section ⇒ stream
serialization regardless of stream flags. Fixed by pinning the source.

2. RMM cross-stream waits (librmm/.../stream_ordered_memory_resource.hpp:401-406):

// "Since we found a block associated with a different stream, we have to
//  insert a wait on the stream's associated event into the allocating stream."
RMM_CUDA_TRY(cudaStreamWaitEvent(stream_event.stream, other_event, 0));

pool_memory_resource<cuda_memory_resource> (the default in
config_device_memory_pool, libcudf-sys/src/operations.cpp:155-164)
derives from stream_ordered_memory_resource. Whenever stream A allocates a
block that stream B last freed, RMM inserts a cudaStreamWaitEvent so A
waits for B. Trace shows ~28 K such waits. By design, but it collapses
cross-stream parallelism.

3. Explicit cudaStreamSynchronize in cuDF (4,736 calls, ~158/iter):

Site

Why

interop/to_arrow_host.cu:361,378,472,543 

every download syncs before returning host data 

groupby/hash/compute_aggregations.cuh:149 

every groupby reads a fallback atomic_flag 

detail/utilities/cuda_memcpy.hpp:87,103 

sync variant of memcpy helper 

detail/utilities/vector_factories.hpp 

many make_*_vector_sync helpers 

Sync-duration histogram: 56% are <5 µs (no-op, GPU already idle but driver
round-trip still costs); ~290 are >500 µs (real waits).

The smoking-gun test

Bench-only Option 1: allocate input column buffers with cudaMallocHost,
wrap via Buffer::from_custom_allocation. Toggled by
CUDF_BENCH_PINNED_INPUT=1 in libcudf-datafusion/benches/aggregate.rs.

cudaPointerGetAttributes confirms the buffer reaches cuDF as type=1
(pinned). ✓

HtoD copies still serialize back-to-back. Total HtoD time unchanged
(2.06 s pageable vs. 2.06 s pinned).

Smaller-N runs segfault: pinned-source cudaMemcpyAsync skips staging, so
the host buffer must outlive the async DMA. Our Arc<PinnedAlloc> was
dropped too early. Pageable is immune because the driver synchronously
stages before returning.

Conclusion: pinning is necessary but not sufficient. Bottlenecks (2) and
(3) re-serialize work even when the copy hardware could run it concurrently.

Hardware ceilings

Even with all software bottlenecks fixed, T4 caps at:

PCIe bandwidth: ~16 GB/s per direction. 320 MB upload/iter ⇒ ~20 ms
minimum.

Copy engines: 2 ⇒ at most 2 transfers physically in flight at once.

So realistic upper bound is ~2-3x upload throughput, not 4x.

Effective throughput today: ~2.6 GB/s. Far below the 16 GB/s ceiling, so
there's real headroom — we're call-overhead and serialization bound, not
bandwidth bound.

PinnedPoolConfig is on the wrong side of the ledger

src/pinned_pool.rs configures cuDF's internal pinned host allocator,
which only governs cuDF's own host allocations (download destinations,
internal scratch). It does not affect the upload-side Arrow source, which is
where 92% of memcpy time lives.

Next steps

Swap the device pool to cuda_async_memory_resource
(librmm/.../cuda_async_memory_resource.hpp). Uses
cudaMallocAsync/cudaFreeAsync — driver-level stream ordering, no
user-space cudaStreamWaitEvent. Small change in operations.cpp.
Re-trace pageable to see if HtoD blocks start overlapping.

Combine with pinned input. This is the run we expect to actually move
wall clock.

NVTX-instrument flush_pending, from_arrow_host_on,
to_arrow_host_on, groupby.aggregate_on to attribute remaining syncs
to specific operations.

Production pin_record_batch in CuDFLoadExec — staging copy with
stream-attached lifetime so the pinned buffer outlives the async DMA.

Realistic gain estimate

If both software bottlenecks are addressed:

~140 ms HtoD/iter → ~40-60 ms (2-3x faster on the upload side)

End-to-end: roughly 1.4-1.6x on this bench shape

Workloads with more kernel compute or larger downloads will benefit more
once the upload serialization is gone.nce the upload serialization is gone.
