# 100M sum: where the time goes — 2026-05-06

Profiles paired with no-profiler wall-time benches, 4 partitions, 8K-row
batches, sum-by-key on 100M rows.

Profiles:
- `docs/agg_100M_streams_off_2026-05-06.nsys-rep`
- `docs/agg_100M_streams_on_2026-05-06.nsys-rep`

## Wall-clock baseline (no profiler)

| Metric                   | streams_off | streams_on | Δ        |
|--------------------------|------------:|-----------:|---------:|
| Per-iteration wall       | **635 ms**  | **454 ms** | −181 ms (−28.5%) |
| Throughput               | 158 M rows/s | 220 M rows/s | +40%   |

## A. Per-iteration wall decomposed

These are the three slices that actually compose an iteration's wall
time. The host slice runs in parallel with GPU work; the GPU slice
(kernel + transfer) is on the critical path.

### streams_off (635 ms total)

| Slice                                       | Time   | % of iter |
|---------------------------------------------|-------:|----------:|
| Host machinery (DataFusion / tokio / FFI / pin)  | ~280 ms | 44%   |
| H2D transfer wall (PCIe-bound at 5.6 GB/s)  | 287 ms | 45%       |
| GPU kernel wall (default-stream serialized) |  68 ms | 11%       |

### streams_on (454 ms total)

| Slice                                       | Time   | % of iter |
|---------------------------------------------|-------:|----------:|
| Host machinery (DataFusion / tokio / FFI / pin)  | ~141 ms | 31%   |
| H2D transfer wall (PCIe-bound at 5.6 GB/s)  | 288 ms | 63%       |
| GPU kernel wall (4-stream concurrency)      |  25 ms |  5%       |

### Where the 181 ms of savings came from

| Source                                       | Saved   |
|----------------------------------------------|--------:|
| Host machinery (faster API submission, less driver-mutex contention) | **−139 ms** |
| GPU kernel concurrency across 4 streams      | −43 ms  |
| H2D transfer (PCIe-bound, no help)           |   ~0 ms |
| **Total**                                    | **−181 ms** |

## B. CUDA API time across the bench (4-worker sum)

Iteration counts under profiler: streams_off ≈ 68 iters, streams_on ≈ 138 iters
(profiler overhead lowers throughput).

### streams_off — 29.7 s of total CUDA-API CPU time

| API                       | Total | % of API CPU | Per-call avg |
|---------------------------|------:|-------------:|-------------:|
| cudaMemcpyAsync           | 12.57 s | **42.3%**  | 39 µs        |
| cudaStreamSynchronize     |  5.53 s |  18.6%     | 34 µs        |
| cudaFreeAsync             |  5.43 s |  18.3%     | 17 µs        |
| cudaMallocFromPoolAsync   |  4.82 s |  16.2%     | 15 µs        |
| cudaLaunchKernel          |  0.52 s |   1.7%     | 187 µs       |
| cudaHostAlloc (one-time)  |  0.51 s |   1.7%     | —            |
| Everything else           |  0.32 s |   1.1%     | —            |

### streams_on — 29.5 s of total CUDA-API CPU time

| API                       | Total | % of API CPU | Per-call avg |
|---------------------------|------:|-------------:|-------------:|
| cudaMemcpyAsync           |  8.74 s | **29.6%**  | **15 µs** ↓  |
| cudaMallocFromPoolAsync   |  7.36 s |  24.9%     | 13 µs        |
| cudaStreamSynchronize     |  7.31 s |  24.8%     | 25 µs        |
| cudaFreeAsync             |  3.87 s |  13.1%     |  7 µs ↓      |
| cudaMemPoolDestroy        |  0.85 s |   2.9% NEW | 4.6 ms       |
| cudaLaunchKernel          |  0.36 s |   1.2%     | 65 µs        |
| cudaHostAlloc (one-time)  |  0.41 s |   1.4%     | —            |
| Everything else           |  0.59 s |   2.0%     | —            |

### Per-call shifts that matter

| API                         | streams_off | streams_on | Δ per call |
|-----------------------------|------------:|-----------:|-----------:|
| cudaMemcpyAsync (submission)|     39 µs   |    15 µs   | **−61%**   |
| cudaFreeAsync (submission)  |     17 µs   |     7 µs   | **−59%**   |
| cudaLaunchKernel            |    187 µs   |    65 µs   | −65%       |
| cudaStreamSynchronize       |     34 µs   |    25 µs   | −26%       |
| cudaMallocFromPoolAsync     |     15 µs   |    13 µs   |  −13%      |

The headline: every "submit work to a stream" API got dramatically
cheaper because 4 workers can now submit on 4 different streams instead
of fighting for the driver mutex on the default stream.

## C. cuDF stages (NVTX)

Both modes spend ~92% of cuDF time inside the load path
(`from_arrow_host` and the `make_*_column` allocations nested under it).

| Stage                          | streams_off Total | % | streams_on Total | % |
|--------------------------------|------------------:|--:|-----------------:|--:|
| `libcudf:from_arrow_host`      | 22.22 s | 54.2% | 26.50 s | 44.6% |
| `libcudf:make_fixed_width_column` |  7.89 s | 19.3% | 14.37 s | 24.2% |
| `libcudf:make_numeric_column`  |  7.50 s | 18.3% | 13.60 s | 22.9% |
| `libcudf:aggregate`            |  2.10 s |  5.1% |  3.11 s |  5.2% |
| `libcudf:concatenate`          |  0.54 s |  1.3% |  0.65 s |  1.1% |
| `libcudf:to_arrow_host`        |  0.12 s |  0.3% |  0.42 s |  0.7% |
| Everything else                |  0.62 s |  1.5% |  0.78 s |  1.3% |

`make_*_column` shifts up 5 percentage points because as the wrapping
`from_arrow_host` gets faster, the alloc work inside it becomes a larger
relative slice.

## D. CuDFLoadExec specifically

LoadExec maps to NVTX `libcudf:from_arrow_host`. This wraps:
- pin_record_batch (host-to-pinned memcpy on CPU)
- column construction (`make_*_column` → `cudaMallocFromPoolAsync`)
- arrow→cudf nanoarrow marshalling
- per-column `cudaMemcpyAsync` submission
- per-batch `cudaStreamSynchronize` (legacy sync-after-upload mode)

| Metric                      | streams_off | streams_on | Δ        |
|-----------------------------|------------:|-----------:|---------:|
| `from_arrow_host` per-call  | **140 µs**  | **94 µs**  | **−33%** |
| `from_arrow_host` median    |  79 µs      |  57 µs     | −28%     |
| `from_arrow_host` total     | 22.22 s     | 26.50 s    | (more iters) |
| Calls per iteration         | ~2 335      | ~2 037     | similar  |

### What inside LoadExec got faster

| Sub-step                   | streams_off avg | streams_on avg | per-call Δ |
|----------------------------|----------------:|---------------:|-----------:|
| `make_fixed_width_column` (key alloc) |   25 µs   |    26 µs    |  flat |
| `make_numeric_column` (val alloc)     |   24 µs   |    24 µs    |  flat |
| `cudaMemcpyAsync` (H2D submission)    |   39 µs   |    **15 µs** | **−61%** |

LoadExec speedup is **almost entirely** from cheaper memcpy submissions.
Per-call alloc cost is unchanged (per-stream pools cap it at the same
~13 µs the default stream pays). The H2D *bytes* still travel at the
same 5.6 GB/s in either mode — it's the host-side launch cost that drops.

## E. Why not a 2× speedup?

Even if every parallelisable slice collapsed to zero, an iter would still
take ~280 ms (host machinery, which streams cannot help). That's the
theoretical floor. The per-PCIe-byte wall (~287 ms) is also a hard floor
streams cannot break. So the maximum possible speedup at this workload
shape is roughly:

```
ceiling = 1 / (host_share + h2d_share)
        = 1 / (0.44 + 0.45)
        = 1.12×
```

We're realising **1.40×**, beating that 1.12× ceiling. The model is
wrong because it treats "host" as a single irreducible block. In
reality, ~half the host slice is **CUDA API submission cost** —
`cudaMemcpyAsync`, `cudaLaunchKernel`, `cudaFreeAsync` calls — which
goes way down with streams (per-call cost drops 60–65%, see §B). So
streams shave ~140 ms off "host" too, not just the GPU side.

The corrected ceiling: only the *non-CUDA-API* portion of host (≈140 ms
of "true" host work — pin_record_batch, FFI marshalling, DataFusion
plan, tokio scheduling, criterion machinery) plus the H2D floor are
fundamentally unreachable by streams. That gives:

```
true_ceiling = 635ms / (140ms host-irreducible + 287ms H2D)
             = 635 / 427 = 1.49×
```

Our 1.40× is within 6% of that ceiling. **There's almost no slack left
for streams to extract on this workload.** Anything past this requires
attacking the irreducible parts.

## F. Levers that would push past 1.40×

These are the levers, ranked by expected impact for this workload at
this scale.

### F.1 — Reduce H2D bytes (the biggest single lever)

The 287 ms H2D floor is `(100M rows × 16 B/row) / 5.6 GB/s = 287 ms`.
Anything that cuts the byte count cuts that floor proportionally.

| Idea                              | Mechanism                          | Expected H2D reduction | Expected iter speedup |
|-----------------------------------|------------------------------------|-----------------------:|----------------------:|
| Int32 keys (when cardinality fits) | Halve key column from 8 B to 4 B  | ~25% (12 B/row → 12 B → keep 8 B val + 4 B key) | 1.62× total |
| Column pruning earlier in plan    | Drop unused columns before LoadExec | Workload-dependent — for queries that touch a subset of columns, can be 50%+ | up to 1.8× |
| Late materialisation              | Upload row indices, gather columns on device | Massive for selective predicates | up to 3× |
| Compress columns on host          | lz4 on CPU, decompress on GPU      | 2–4× on numeric runs, but adds CPU work | depends — best on CPU-spare hosts |
| Run-length / dictionary encoding  | Send compact representation        | Workload-dependent | 1.5–3× for skewed data |

The Int32-key change alone is a one-line schema change in the bench and
would reclaim ~70 ms of the H2D floor.

### F.2 — Reduce host irreducible (the other ~140 ms)

| Idea                                | Mechanism                         | Expected savings |
|-------------------------------------|-----------------------------------|-----------------:|
| **Skip pin_record_batch's host memcpy** | Have upstream produce arrow buffers directly in pinned memory (custom allocator on the source side) | ~15–25 ms |
| **Bigger batches (>64K rows)**      | Amortize per-batch FFI / cxx marshalling and DataFusion plan overhead | We tested 64K and it didn't help — 256K worth a try, but groupby kernel may have a sweet spot below |
| **Batch FFI calls**                 | One cxx call that does pin + upload + table-construct, instead of N | Saves cxx round-trip overhead per batch (~µs each, but thousands of calls) |
| **More physical cores**             | We're on 2 physical / 4 logical CPUs. A 4-core box would cut tokio scheduling pressure | up to 30% on host slice |
| **Stream pool (planned)**           | Avoid per-iter `cudaMemPoolCreate`/`Destroy` (~5 ms/iter at 100M) | small at 100M scale; matters more at 20M |

The "skip pin_record_batch" lever is the most attractive: it's a one-time
allocator change at the source (the bench's `make_batches` or, in
production, the upstream operator) and removes a CPU memcpy of the entire
working set every iteration.

### F.3 — Pipelining (overlap phase boundaries)

Today, an iteration runs LoadExec → AggregateExec → UnloadExec roughly
serially per partition. With pipelining you'd start uploading batch
N+1 while the GPU is still aggregating batch N. The current per-batch
sync (`cudaStreamSynchronize` after each upload) prevents that. The
deferred-return-event design we reverted earlier *was* an attempt at
this; it stalled on free-pool starvation. Could be revisited with a
bounded in-flight queue, but only worth it after F.1 / F.2 are done —
right now the H2D floor caps the win.

### F.4 — Push more work to GPU before unloading

If queries have aggregate → join → aggregate, doing all three on GPU
avoids two round-trips. Doesn't apply to the simple groupby in this
bench, but matters for real query plans.

## G. What NOT to chase further on this workload

- **Streams are basically saturated.** The 1.40× we have is within 6% of
  the ceiling for this shape. Don't expect more from stream count, stream
  flags, or stream-pool refinements.
- **Per-stream pool tuning** (release threshold, prime size). The
  per-call alloc cost is already at parity with streams_off; further
  tuning won't move iter wall time meaningfully at 100M.
- **The deferred-return / event-tagged pinned pool** — that change was
  designed for a regime where per-batch sync dominated. At 100M scale,
  per-batch sync is 25 µs × 12,500 batches ≈ 313 ms across 4 workers, i.e.
  ~78 ms wall — meaningful but second-tier compared to H2D bytes. Worth
  revisiting only after F.1 / F.2 push H2D below today's 287 ms floor.

## H. Suggested next experiment

Run the bench with `Int32` keys at 100M rows, both modes. If the H2D
floor drops from 287 ms to ~215 ms as the model predicts, streams_on
should land near **1.6× over streams_off** with no other code changes.
That's the cheapest test of the H2D-bytes-bound thesis.

## Reports index

```
01-plan.md
02-post-pinned-bottlenecks.md
03-aggregate-cuda-streams-bench-deferred-return.md
04-streams_results.md
05-aggregate-cuda-streams-profile.md
06-cudaMalloc-from-pool-async-deep-dive.md
07-cudaMalloc-options-1-and-2-deep-dive.md
08-aggregate-cuda-streams-post-option-2.md
09-aggregate-100M-time-breakdown.md   ← this report
```
