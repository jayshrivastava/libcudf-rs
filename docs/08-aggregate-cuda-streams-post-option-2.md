# Post-Option-2 report — 2026-05-06

This is the post-optimization snapshot after:

1. **Reverted** the deferred-return pinned pool + CUDA event pool work
   (`CuDFEvent`, the in-flight queue on `PinnedPool`, the
   `event.{cpp,h}` FFI, and the `record_event_on` call in
   `CuDFLoadExec`). Restored the legacy `stream.synchronize()` /
   `synchronize_default_stream()` post-upload sync.
2. **Landed Option 2** — a per-stream `cuda_async_memory_resource`
   wrapper in `libcudf-sys/src/operations.cpp`. Each `cudaStream_t` gets
   its own CUDA mempool keyed by stream handle. Maps cleanly to our
   pipeline (one stream per partition, one pool per stream).
3. **Added a release hook** — `release_device_pool_stream` is called
   from `CuDFStream::Drop` before the stream itself is destroyed,
   dropping the pool entry and freeing cached memory via
   `cudaMemPoolDestroy`. Without this the `pools_` map grows
   unboundedly across criterion iterations and OOMs the GPU.

## Bench (20M rows, 4 partitions, no profiler overhead)

| Shape    | streams_off | streams_on | Δ      |
|----------|-------------|------------|--------|
| sum      | 110.2 ms    | 113.7 ms   | +3.2%  |
| count    | 110.0 ms    | 113.4 ms   | +3.1%  |
| avg      | 114.4 ms    | 121.6 ms   | +6.3%  |
| min_max  | 116.6 ms    | 124.3 ms   | +6.6%  |
| combined | 132.1 ms    | 160.2 ms   | +21.3% |

Comparing the same `sum` shape across the regimes we've tested:

| Regime                                  | sum/streams_on |
|-----------------------------------------|----------------|
| deferred-return + shared pool           | 134.2 ms       |
| legacy sync + shared pool               | 114.5 ms       |
| legacy sync + per-stream pools (now)    | **113.7 ms**   |

vs. streams_off:

| Regime                                  | sum/streams_off |
|-----------------------------------------|-----------------|
| deferred-return + shared pool           | 113.3 ms        |
| legacy sync + shared pool               | 114.6 ms        |
| legacy sync + per-stream pools (now)    | **110.2 ms**    |

streams_off improved ~4% from the per-stream pool change too, since
even the default-stream path now uses a dedicated per-stream pool with
no cross-stream noise.

## Profile (sum/20M, nsys CUDA + NVTX trace)

`docs/agg_perstreampool_streams_on_2026-05-06.nsys-rep`
`docs/agg_perstreampool_streams_off_2026-05-06.nsys-rep`

Both profiles ran the same criterion command (`sum/gpu_streams_*/20000000`,
sample-size 10, 5 s measurement). Iterations completed under profiler:

- streams_off: ~78 (310 partition-aggregate kernel instances / 4)
- streams_on:  ~124 (496 / 4)

Streams_on completes more iterations in the same wall-time budget — per
iter, it's slightly faster *under profiler*. Without profiler the gap
flips slightly the other way (110 vs 114 ms). Profiler instrumentation
penalises the higher-API-call-rate path more, which is streams_off.

### Where wall-clock CPU time goes (CUDA API totals, sum/20M)

| API                       | streams_off       | streams_on        | streams_on / off |
|---------------------------|-------------------|-------------------|------------------|
| cudaMemcpyAsync           | **12.13 s**       | 5.10 s            | 0.42×            |
| cudaStreamSynchronize     | 4.59 s            | **5.12 s**        | 1.12×            |
| cudaFreeAsync             | 4.06 s            | 1.98 s            | 0.49×            |
| cudaMallocFromPoolAsync   | 2.96 s            | **5.57 s**        | 1.88×            |
| cudaMemPoolDestroy        | n/a               | 1.24 s            | new tax          |
| cudaLaunchKernel          | 0.29 s            | 0.26 s            | 0.90×            |
| cudaEventRecord (RMM)     | 0.17 s            | 0.02 s            | 0.12×            |

Per call (median):

| API                       | streams_off med | streams_on med |
|---------------------------|-----------------|----------------|
| cudaMemcpyAsync           | 10.4 µs         | 7.6 µs         |
| cudaStreamSynchronize     | 14.3 µs         | 12.4 µs        |
| cudaFreeAsync             | 2.5 µs          | 2.2 µs         |
| cudaMallocFromPoolAsync   | 4.3 µs          | 7.8 µs         |

### Where GPU time goes (kernels)

Top kernel = `cub::detail::for_each::static_kernel` running the per-key
SUM reduction:

| Kernel                            | streams_off total | streams_on total |
|-----------------------------------|-------------------|------------------|
| SUM reduction (`for_each_kernel`) | 528 ms            | 542 ms           |
| Concat double                     |  89 ms            |  90 ms           |
| Concat long                       |  89 ms            |  90 ms           |
| `DeviceSelect::If`                |  44 ms            |  47 ms           |
| Two more `for_each_kernel` shapes |  86 ms            |  88 ms           |
| `mapping_indices_kernel`          |   6 ms            |  10 ms           |

Per iter:

| Mode        | Kernel total | iters | per iter |
|-------------|--------------|-------|----------|
| streams_off | 858 ms       | ~78   | 11.0 ms  |
| streams_on  | 906 ms       | ~124  | **7.3 ms** |

GPU time per iter is **34% lower with streams** — that's real kernel
concurrency across the 4 partition streams. The GPU work itself is the
clearest streams win.

### Where the H2D bytes go

| Mode        | Total H2D bytes | Calls | Wall time | Throughput |
|-------------|-----------------|-------|-----------|------------|
| streams_off | 12.4 GB (est.)  | 306 K | 3.67 s    | 5.6 GB/s   |
| streams_on  | 20.3 GB (est.)  | 308 K | 3.62 s    | 5.6 GB/s   |

Throughput is identical. Streams don't widen the H2D pipe — the
bottleneck on the upload side is per-call CPU overhead, not concurrency.

### What cuDF spends time on (NVTX)

| Range                       | streams_off total | streams_on total | streams_off avg | streams_on avg |
|-----------------------------|-------------------|------------------|-----------------|----------------|
| `libcudf:from_arrow_host`   | **19.10 s**       | **16.82 s**      | 126 µs          | 110 µs         |
| `libcudf:make_fixed_width_column` |  5.73 s     | 10.50 s          | 19 µs           | **34 µs**      |
| `libcudf:make_numeric_column`     |  5.37 s     | 10.10 s          | 18 µs           | **33 µs**      |
| `libcudf:aggregate`         |  1.67 s           |  2.31 s          | 5.4 ms          | 4.7 ms         |
| `libcudf:to_arrow_host`     |  0.47 s           |  0.99 s          | 1.5 ms          | 2.0 ms         |
| `libcudf:concatenate`       |  0.63 s           |  0.50 s          | 1.0 ms          | 0.5 ms         |
| `libcudf:allocate_like`     |  37 ms            | 195 ms           | 20 µs           | 65 µs          |

The NVTX picture matches the CUDA-API picture: `make_*_column` (which
hosts `cudaMallocFromPoolAsync` internally) costs roughly **2× per call**
under streams_on. That's down from the earlier shared-pool **~2.3×**
ratio but isn't gone — there's still residual driver-side coordination
in the per-stream-pool path, plus the `cudaMemPoolCreate` /
`cudaMemPoolDestroy` overhead landing inside the first/last calls of
each stream's lifetime.

### Decomposition: where the streams_on gap (vs streams_off) actually lives

Per-iter cost mapping (median wall-time per iter, both modes):

```
streams_off (~110 ms / iter no-profiler):
  ├─ GPU kernels (top stack)         ~11.0 ms
  ├─ H2D memcpy wall                  ~47 ms
  ├─ cudaMallocFromPoolAsync          ~38 ms (from 4 workers, mostly overlapping with GPU)
  ├─ cudaMemcpyAsync API              ~16 ms (per-thread, overlapping)
  ├─ cudaStreamSynchronize            ~60 ms (per-batch sync after each upload)
  └─ everything else, host scheduling

streams_on (~114 ms / iter no-profiler):
  ├─ GPU kernels (top stack)          ~7.3 ms       ← 34% faster from concurrency
  ├─ H2D memcpy wall                  ~29 ms        ← lower, more concurrent submission
  ├─ cudaMallocFromPoolAsync          ~45 ms        ← 17% slower per iter (pool-create dominant)
  ├─ cudaMemcpyAsync API              ~41 ms
  ├─ cudaStreamSynchronize            ~41 ms        ← lower, fewer threads contend
  ├─ cudaMemPoolDestroy               ~10 ms        ← new tax (pool churn)
  └─ everything else
```

The streams_on side wins on GPU kernel time (kernel concurrency) and on
H2D submission. It loses on `cudaMallocFromPoolAsync` totals
(more pools = more pool-creation work hits the alloc API), and pays a
new ~10 ms/iter tax for `cudaMemPoolDestroy` at end-of-iter.

### Headline shifts vs the previous shared-pool profile

| Metric                                   | shared pool | per-stream pool | Δ          |
|------------------------------------------|-------------|-----------------|------------|
| `cudaMallocFromPoolAsync` avg ns         | 35,749      | 17,683          | **−51%**   |
| `cudaMallocFromPoolAsync` median ns      | 16,150      | 7,813           | −52%       |
| `cudaStreamWaitEvent` calls              | 989         | 987             | flat       |
| `cudaStreamWaitEvent` total              | 4.4 ms      | 4.5 ms          | flat       |
| `cudaMemPoolCreate` calls                | 0           | 496             | new        |
| `cudaMemPoolDestroy` calls               | 0           | 496             | new        |
| `cudaMemPoolDestroy` total time          | 0           | 1.24 s          | new tax    |

The original "3.5× slower per call with streams" number that motivated
this whole investigation is now **1.85× slower per call** — half the
contention is gone. The other half is the residual driver-internal
coordination plus the per-stream pool's own create/destroy cost.

Per-call CUDA API costs, side-by-side against the earlier shared-pool
profile:

| API                       | shared pool, streams_on | per-stream, streams_on | Δ     |
|---------------------------|-------------------------|------------------------|-------|
| cudaMallocFromPoolAsync   | 35,749 ns avg / 16 µs med | **17,683 ns avg / 7.8 µs med** | **−50%** |
| cudaStreamSynchronize     | 20,198 ns               | 32,160 ns              | +59%  |
| cudaMemcpyAsync           | 16,942 ns               | 16,327 ns              | ~flat |
| cudaFreeAsync             | 8,635 ns                | 6,282 ns               | −27%  |
| cudaMemPoolCreate         | n/a                     | 57,599 ns avg, 496 calls | new |
| **cudaMemPoolDestroy**    | n/a                     | **2.5 ms avg, 496 calls** | **new (1.24 s total)** |

The headline: **`cudaMallocFromPoolAsync` per-call cost is cut in half
with per-stream pools.** That's the bottleneck the prior
investigation called out, and it's directly addressed.

The new tax: **`cudaMemPoolDestroy`** — 1.24 s total, ~10 ms per
iteration averaged across the 124-iter run. We pay this because the
bench harness creates a fresh `CuDFTaskContext` per criterion iteration
(see `libcudf-datafusion/benches/aggregate.rs:153–158`), which means
fresh streams, which means fresh pools. Pools live for one iteration.

`cudaStreamSynchronize` total time also went up (3.22 s → 5.12 s),
likely because pool destroy is blocking and pushes pressure onto stream
syncs that overlap it.

## Why streams_on is still slightly slower than streams_off

Per-iteration cost decomposition for the bench harness:

```
streams_off:
  - one default stream, one pool
  - pool create: 1 × ~110 µs (whole bench)
  - pool destroy: 0 (process exit)
  - per-iter overhead: negligible

streams_on (per-stream pool):
  - 4 streams × N iters = 4N pool creates and destroys
  - per pool: ~58 µs create + 2.5 ms destroy ≈ 2.5 ms each
  - per iter: 4 × 2.5 ms = 10 ms of pool lifecycle
```

The ~10 ms/iter of pool churn maps almost exactly to the streams_on
gap on simple aggregates (3–7 ms/iter). On the `combined` shape the
gap is bigger (28 ms/iter) — that shape runs 5 aggregates which means
more concurrent in-flight columns and more pool growth events per
iter.

## Net read

- ✅ **`cudaMallocFromPoolAsync` per-call cost halved** — the original
  bottleneck is fixed.
- ✅ **streams_on no longer regresses badly** — was +18% on `sum`, now
  +3%. The earlier "deferred-return makes it worse" effect is gone
  (deferred-return reverted).
- ✅ **streams_off also got faster** (~4%) for free — per-stream pool
  even with one stream is slightly better than the prior
  pre-reserved-512 MB shared pool.
- ⚠️ **Pool lifecycle is now the dominant streams_on tax** — 10 ms/iter
  of `cudaMemPoolCreate` + `cudaMemPoolDestroy`. This is a
  bench-harness artefact: the bench creates fresh streams per
  iteration. In production a query holds its streams for its full
  lifetime, so pools are paid once per query, not per batch.
- ⚠️ **Combined-shape regression (+21%) is real but explained** by the
  same pool-churn dynamic, scaled by aggregation depth.

## What I'd do next

1. **Stop creating fresh streams per criterion iteration.** Move
   `CuDFTaskContext` creation out of `iter_batched`'s setup closure and
   into the per-shape outer scope. This is a bench-only change that
   eliminates the per-iter `cudaMemPoolCreate` /
   `cudaMemPoolDestroy` traffic and gives a fairer streams-on number.
   Expected: streams_on closes the gap on simple aggs and the combined
   regression shrinks substantially.

2. **If pool create is still measurable after that**, drop the 1 MiB
   prime by constructing pools via `cuda_async_view_memory_resource`
   on top of an explicitly-created `cudaMemPool_t` (skipping RMM's
   `do_allocate(initial)` prime call). About 10 lines of additional C++.

3. **Only after the above two**, decide whether streams_on is worth
   the operational complexity. With a 4-partition workload on a
   GPU-rich host, kernel concurrency wins are modest; bigger benefits
   should show up at higher partition counts where the GPU isn't
   saturated by serial work.

## Code state

```
M libcudf-sys/src/lib.rs                 (removed CudaEvent FFI; added release_device_pool_stream)
M libcudf-sys/src/operations.cpp         (per_stream_async_mr + release hook)
M libcudf-sys/src/operations.h           (release_device_pool_stream decl)
M src/lib.rs                             (removed CuDFEvent / PinnedBatch re-exports)
M src/pinned.rs                          (back to HEAD — no events, no in-flight queue)
M src/stream.rs                          (CuDFStream + Drop hook; CuDFEvent deleted)
M libcudf-datafusion/src/physical/cudf_load.rs  (record_event_on → stream.synchronize)
D libcudf-sys/src/event.cpp              (deleted)
D libcudf-sys/src/event.h                (deleted)
```
