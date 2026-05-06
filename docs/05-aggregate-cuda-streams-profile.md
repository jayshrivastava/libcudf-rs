# Aggregate streams_on vs streams_off profile — 2026-05-06

**Question:** in legacy (per-batch sync) mode, where do CUDA streams help and where don't they? `LIBCUDF_LEGACY_PINNED=1` was used to remove deferred-return as a confound — both runs use synchronize-per-upload, so the only variable is per-partition streams.

**Setup:** 20M rows, 4 partitions, single SUM-by-key, `cuda_async_memory_resource` device pool, pinned input enabled.

**Profiles** (CUDA + NVTX trace via nsys):

- `docs/agg_legacy_streams_on_2026-05-06.nsys-rep`
- `docs/agg_legacy_streams_off_2026-05-06.nsys-rep`

## Wall time (criterion, no profiler)

20M rows, legacy mode:

| Shape    | streams_off | streams_on | Δ      |
|----------|-------------|------------|--------|
| sum      | 114.6 ms    | 114.5 ms   |  ~0%   |
| count    | 111.3 ms    | 106.2 ms   | −4.6%  |
| avg      | 116.8 ms    | 104.9 ms   | −10.2% |
| min_max  | 115.6 ms    | 122.2 ms   | +5.7%  |
| combined | 133.4 ms    | 133.9 ms   | +0.4%  |

Streams roughly break even on average across the shape mix. Wins on count/avg, loss on min_max, neutral on sum/combined.

## Per-call CUDA API submission cost (sum/20M, profile, avg ns)

| API                       | streams_on | streams_off | Δ              |
|---------------------------|------------|-------------|----------------|
| cudaMallocFromPoolAsync   | 35,749     | 10,292      | **+247%**      |
| cudaMemcpyAsync           | 16,942     | 41,993      | −60%           |
| cudaStreamSynchronize     | 20,198     | 29,612      | −32%           |
| cudaFreeAsync             |  8,634     | 14,320      | −40%           |
| cudaLaunchKernel          | 36,336     | 160,894     | −77%           |

(Per-call totals divided by call count; with N streams, 4 worker threads can submit concurrently — that's why per-call cost drops for most APIs.)

## GPU kernel time per iteration (top kernel = SUM `for_each::static_kernel`)

| Mode        | Kernel total | ~iters | Per iter |
|-------------|--------------|--------|----------|
| streams_on  | 535 ms       | ~124   | **4.3 ms** |
| streams_off | 314 ms       | ~46    | 6.8 ms   |

The dominant kernel is 37% shorter per iteration with streams — that's real concurrency, partitions overlapping on the GPU.

## H2D bandwidth

| Mode        | Total bytes | Total time | Throughput |
|-------------|-------------|------------|------------|
| streams_on  | 20.3 GB     | 3.65 s     | 5.6 GB/s   |
| streams_off | 12.1 GB     | 2.16 s     | 5.6 GB/s   |

Identical effective throughput. Streams don't speed up the H2D path at all — bandwidth here is bottlenecked by per-call overhead and memcpy size (~66 KB avg), not by lack of pipeline parallelism.

## Where streams help

1. **Kernel concurrency is real.** The dominant SUM reduction kernel is 37% faster per iteration when 4 partitions execute on 4 streams. The GPU has plenty of SMs sitting idle in the streams_off case.
2. **Concurrent driver submission.** Memcpy, kernel launch, free, and stream-sync API calls are all 30–80% cheaper per call when 4 worker threads submit on independent streams. Less time blocked on the driver mutex.

## Where streams don't help (or hurt)

1. **Async allocator becomes a hot-spot.** `cudaMallocFromPoolAsync` is **3.5× slower per call** with streams (avg 36 µs vs 10 µs, max 21 ms with streams). Even with `cuda_async_memory_resource`, the per-stream allocation tracking and any cross-stream sharing has real overhead. This eats most of the kernel-concurrency win.
2. **Stream lifecycle isn't free.** The profile run created/destroyed 496 streams; `cudaStreamDestroy` totaled 224 ms (avg 0.45 ms, max 10 ms). Re-use across iterations would amortize this, but currently we allocate a fresh `CuDFTaskContext` per iteration so per-(segment, partition) streams are short-lived.
3. **RMM internal events.** 495 `cudaEventCreateWithFlags` calls (49 ms total, 99 µs avg) — these are RMM tracking allocation lifetimes for stream-ordered free, not our pool. They cost real time per allocation.
4. **H2D throughput is unchanged.** Streams don't widen the upload pipe; the bottleneck is per-batch CPU overhead, which streams don't reduce, and total H2D time scales linearly with iterations.

## Net read

In legacy (per-batch sync) mode the kernel-concurrency win and the allocator/lifecycle cost roughly cancel for `sum`. On simple aggregates (`count`, `avg`) where the kernel is the dominant cost, streams win 5–10%. On `min_max` (where there are more allocations per iteration — separate min and max columns to manage) the allocator pressure dominates and streams lose 6%.

The right next move depends on which ceiling you want to lift:
- **Lower allocator overhead.** Pre-warm or per-stream-pinned the RMM async pool, or feed cuDF an arena-style allocator so per-batch `cudaMallocFromPoolAsync` calls drop. This would unlock more of the kernel-concurrency win.
- **Reduce stream lifecycle cost.** Cache `CuDFStream` instances across iterations rather than allocating fresh per `CuDFTaskContext`. Saves the 224 ms of stream destroy time and the matching create time.
- **Larger batches.** At 8K rows × ~66 KB per memcpy, the H2D path is launch-bound. Bigger batches would mean fewer (memcpy + launch + alloc) cycles and amortize the per-call costs that streams already make cheaper.

## Aside: deferred-return surprise

For `sum/20M/streams_on`, **legacy is faster than deferred-return**: 114.5 ms vs 134 ms. That's the opposite of what the deferred-return change was supposed to deliver.

Working theory: the in-flight queue holds buffers until their event fires. If the GPU can't keep up with the host's allocation rate, the free pool stays empty and `PinnedHostBuffer::new` falls through to `cudaMallocHost` (~500 µs each). Per-batch sync forces the source to be available immediately, so the free pool churns tightly and we never `cudaMallocHost` mid-iteration.

This is a separate investigation — we could verify by counting `cudaMallocHost` calls in a deferred-return profile and adding a fixed-size in-flight cap (or eager-drain when free is below threshold).
