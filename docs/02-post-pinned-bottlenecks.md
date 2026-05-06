# Aggregate Bench: What's Left After Pinning

Status report after wiring pinned-host staging into `CuDFLoadExec`. Source
profile: `agg_pinned_pool.nsys-rep`, bench shape `combined/gpu/20000000`,
55 iterations, default partitioning, no explicit CUDA streams.

## Result vs. baseline

Apples-to-apples Criterion comparison (`combined/gpu`):

| Rows | Pageable | Pinned + pool | Speedup |
| ---: | ---: | ---: | ---: |
| 1M | 37.25 ms | 35.49 ms | 1.05× |
| 5M | 63.05 ms | 54.61 ms | 1.15× |
| 20M | 152.24 ms | 123.58 ms | **1.23×** |

Across other shapes at 20M (sum / count / avg / combined): consistent
1.28-1.33× speedup.

## Are HtoD copies still serializing?

**Yes — by design, because everything still runs on the CUDA default stream.**
Pinning didn't change that and was never meant to. What pinning *did* change
is that the CPU is no longer blocked during each upload.

Mid-run timeline of consecutive HtoDs (excerpt):

```text
   start_us      end_us  dur_us gap_to_prev_us stream
 3932506.61  3932612.40  105.79          —      7
 3932613.39  3932719.28  105.89        0.99     7
 3932921.52  3933027.37  105.86      202.24     7   ← gap (kernel ran here)
 3933028.37  3933134.25  105.89        0.99     7
 3933135.24  3933241.07  105.82        0.99     7
 3933242.06  3933347.91  105.86        0.99     7
 ...
```

Observations:

- **Only one stream** (id=7 = default).
- **Copies are back-to-back** with ~1 µs gaps when the stream is uploading
  (good — no idle time between transfers).
- The occasional ~200 µs gap is when a kernel runs on the same stream
  (single-stream serialization).
- HtoD copy duration is ~106 µs each — the same as pageable. Pinning saves
  CPU staging time, not GPU DMA time.

To get *real* parallelism (two HtoDs in flight simultaneously on T4's two
copy engines, or a copy and a kernel overlapping), independent partitions
need their own non-default streams. That's a separate change.

## Per-iteration time budget

55 iterations of `combined/20M`, ~127 ms each (~7.0 s total bench wall):

```text
                   per-iter
GPU work:
  HtoD memcpy       70.7 ms   ────────────────────────  56%
  Kernels           37.4 ms   ──────────────             29%
  DtoH memcpy        5.2 ms   ──                          4%
  DtoD memcpy        0.5 ms                              <1%
  ─────────────────────────
  Total GPU       ≈113.8 ms                              90%

Wall-clock         127 ms                              100%
Gap (CPU work)    ≈ 13 ms                              ≈10%
```

The GPU is busy ~90% of the wall clock. The remaining ~10% is CPU work
that doesn't overlap (DataFusion plan execution, kernel launches,
`pin_record_batch`'s per-batch CPU memcpy from arrow → pinned, sync waits).

## What pinning actually fixed

Side-by-side from the same trace pair (`agg_baseline.nsys-rep` vs.
`agg_pinned_pool.nsys-rep`, totals across the bench run):

| API call | Pageable | Pinned + pool |
| --- | ---: | ---: |
| `cudaMemcpyAsync` total | 22.1 s (80% API) | 7.0 s (41% API) |
| `cudaMemcpyAsync` avg/call | **421 µs** | **118 µs** |
| `cudaStreamSynchronize` total | 1.5 s (6%) | 7.4 s (43%) |
| HtoD GPU time per copy | 106 µs | 106 µs |

The smoking gun is the per-call `cudaMemcpyAsync` cost dropping from
421 µs → 118 µs. That ~300 µs/call we recovered is the CPU staging step
that pageable does *inside* `cudaMemcpyAsync` before the call returns.
Pinned-source `cudaMemcpyAsync` just enqueues the DMA and returns.

## Where the remaining time goes

### Top GPU kernels (per iter)

| Kernel | Per iter | % of kernel time | Notes |
| --- | ---: | ---: | --- |
| `cub::for_each::static_kernel` | 32 ms | 87% | dominant — cuDF's bulk per-element worker, used by concat / reductions |
| `fused_concatenate_kernel` | 3.4 ms | 9% | column concat (Int64 + Float64) |
| `DeviceSelectSweepKernel` | 0.95 ms | 3% | hash-table fill / select |
| `transform_kernel` | 0.5 ms | 1% | small per-element transforms |
| `mapping_indices_kernel` | 0.2 ms | <1% | groupby hash mapping |

Most kernel time is generic CUB `for_each` instances — these come from
column concat and the per-row work cuDF does inside the groupby. Not
trivially avoidable from outside cuDF.

### CPU-side bottleneck shift

`cudaStreamSynchronize` is now the largest single API cost (7.4 s total,
27,300 calls, ~270 µs avg). Sources, by approximate count:

- ~320 calls/iter: our explicit `synchronize_default_stream()` after each
  load batch (so the pinned source can be safely freed).
- ~140 calls/iter: cuDF internal syncs (`to_arrow_host`,
  `compute_aggregations.cuh:149`, `vector_factories` `_sync` helpers).

Most of these overlap with GPU work (they finish quickly because the GPU
has already drained), so they don't translate 1:1 into wall-clock cost.
Estimated wall-clock cost of our explicit syncs: 5-15 ms/iter.

## Options for the next 10-30%

In rough order of expected lift × effort:

1. **Defer load-side syncs.** Hold the pinned batch in a small ring (e.g. 8
   deep) and sync only when the ring is full or the partition stream ends.
   The natural cuDF sync points (groupby, unload) cover most of the
   correctness window, so we'd issue ~30 syncs/iter instead of 320. Estimated
   1.05-1.10× on top of current.
2. **Multi-stream loads.** Give each partition its own non-default stream.
   Two copy engines on T4 means up to 2 HtoDs can be in flight at once.
   Estimated 1.3-1.5× upload throughput, but only if RMM's stream-ordered
   allocator doesn't re-serialize via `cudaStreamWaitEvent` (see
   `docs/plan.md` for that issue).
3. **Larger / fewer batches.** Per-copy overhead is fixed at ~20 µs of the
   current 106 µs, so doubling batch size (640 KB → 1.28 MB) would cut total
   HtoD count in half and shave ~10 ms/iter at 20M. Easy if we're willing to
   change `target_batch_size`.
4. **Skip our extra CPU memcpy.** `pin_record_batch` does an arrow → pinned
   memcpy on the host before the DMA. If the arrow source were itself pinned
   (custom allocator upstream, parquet decompressor output, etc.), we could
   skip that copy. Wider change but fundamentally faster than what we have.

## TL;DR

- Pinning landed cleanly. **1.23× faster at 20M; 1.28-1.33× across other
  shapes** with full test pass.
- The GPU is now ~90% busy each iteration. HtoD memcpy on the default
  stream is 56% of wall.
- "Still serializing" = yes, but only because everything is on one CUDA
  stream. That's the natural next lever.
- The CPU was the previous bottleneck (pageable's hidden staging). It isn't
  any more — `cudaMemcpyAsync` per-call cost dropped 3.5× (421 µs → 118 µs).
