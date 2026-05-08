# TPCH Q1 p1 bs65536 Breakdown

Dataset and command shape:

```bash
./target/release/dfbench run \
  --gpu --dataset tpch_sf10 -q q1 -n 1 -i 3 \
  --batch-size 65536 \
  --no-compare --no-store --debug
```

## Baseline: `LIBCUDF_PINNED_UPLOAD_REAPER=0`

Artifacts:

- Debug log: `/tmp/tpch_q1_p1_bs65536_main_debug.log`
- Nsight report: `/tmp/libcudf-q1-profiles/q1_p1_bs65536_main_nsys.nsys-rep`
- Nsight SQLite: `/tmp/libcudf-q1-profiles/q1_p1_bs65536_main_nsys.sqlite`

Wall time:

- Iteration 0: `5187.8 ms`
- Iteration 1: `4155.8 ms`
- Iteration 2: `4225.7 ms`
- Steady average, iterations 1-2: `4190.8 ms`

Steady-state operator metrics, averaged over iterations 1-2. These are not
exclusive wall-time buckets because the physical plan pipelines batches.

| Area | Time | Approx. Wall % | Notes |
|---|---:|---:|---|
| `CuDFLoadExec` elapsed compute | `2230.0 ms` | `53.2%` | Host-to-cuDF path dominates |
| DataSource `time_elapsed_processing` | `1390.0 ms` | `33.2%` | Parquet scan/decode work |
| `CuDFLoadExec input_wait_time` | `1635.0 ms` | `39.0%` | Waiting on upstream source batches |
| `CuDFLoadExec cast_time` | `823.5 ms` | `19.7%` | Arrow type normalization/cast |
| `CuDFLoadExec pin_time` | `508.4 ms` | `12.1%` | Copy into pinned host buffers |
| `CuDFLoadExec import_time` | `137.2 ms` | `3.3%` | cuDF import enqueue/bridge |
| `CuDFLoadExec sync_time` | `719.3 ms` | `17.2%` | Default-stream H2D completion wait |
| `CuDFLoadExec output_send_time` | `300.1 ms` | `7.2%` | Channel/downstream pacing |
| `CuDFFilterExec` elapsed compute | `250.7 ms` | `6.0%` | GPU filter |
| GPU expression projection | `59.7 ms` | `1.4%` | Projection below aggregate |
| `CuDFAggregateExec` elapsed compute | `592.3 ms` | `14.1%` | GPU groupby/aggregate |

Nsight Systems warmup plus one measured execution:

- Measured iteration wall time: `4238.5 ms`
- H2D GPU memcpy time: `1.71 s` across warmup+measured, roughly `0.85 s/query`
- GPU kernel time: `1.01 s` across warmup+measured, roughly `0.50 s/query`
- Top kernel: groupby hash aggregation, `562 ms` across warmup+measured, roughly `281 ms/query`
- CUDA API wait is dominated by `cudaStreamSynchronize`: `2.37 s` across warmup+measured

### `cudaStreamSynchronize` Attribution

Nsight CPU backtraces were not available in this environment because CPU
profiling was disabled by the system configuration. Attribution below is based
on CUDA API timestamps, thread names, and containment in cuDF NVTX ranges.

Totals are from the Nsight capture, which includes warmup plus one measured
execution. Divide by roughly two for a per-query number.

| Source | Calls | Time | Approx. per query | Notes |
|---|---:|---:|---:|---|
| Explicit `CuDFLoadExec` post-upload fence | `1,856` | `1,326.3 ms` | `663.2 ms` | First sync after each `from_arrow_host`; one per input batch |
| Pinned buffer `allocate_sync` bookkeeping | `24,128` | `142.1 ms` | `71.0 ms` | 13 pinned Arrow buffer allocations per input batch |
| cuDF aggregate NVTX range | `1,326` | `577.2 ms` | `288.6 ms` | Groupby/aggregate synchronization |
| cuDF filter/copy_if NVTX ranges | `25,984` | `109.5 ms` | `54.8 ms` | `copy_if` plus nested `thrust::copy_if` |
| cuDF concatenate/string setup NVTX ranges | `1,760` | `87.0 ms` | `43.5 ms` | `concatenate` plus `create_strings_device_views` |
| cuDF binary operation NVTX range | `11,256` | `42.3 ms` | `21.2 ms` | GPU expression work |
| Other / small NVTX or outside active NVTX ranges | `29,898` | `85.6 ms` | `42.8 ms` | Mostly small syncs |

Relevant call paths:

- Explicit load fence: `libcudf-datafusion/src/physical/cudf_load.rs` calls
  `synchronize_default_stream()` after `CuDFTable::from_arrow_host(...)`.
  That calls `cuda_default_stream_synchronize()`, whose C++ bridge calls
  `cudf::get_default_stream().synchronize()`.
- Pinned allocation syncs: `src/pinned.rs` allocates every copied Arrow buffer
  with `pinned_mr().allocate_sync(bytes)`. RMM's `device_memory_resource`
  `allocate_sync` calls `stream.synchronize()` after allocation, and
  `rmm::cuda_stream_view::synchronize()` calls `cudaStreamSynchronize()`.

## Reaper Enabled: `LIBCUDF_PINNED_UPLOAD_REAPER=1`

Artifacts:

- Debug log: `/tmp/tpch_q1_p1_bs65536_reaper_debug.log`
- Nsight report: `/tmp/libcudf-q1-profiles/q1_p1_bs65536_reaper_nsys.nsys-rep`
- Nsight SQLite: `/tmp/libcudf-q1-profiles/q1_p1_bs65536_reaper_nsys.sqlite`

Wall time:

- Iteration 0: `5320.2 ms`
- Iteration 1: `4276.4 ms`
- Iteration 2: `4278.2 ms`
- Steady average, iterations 1-2: `4277.3 ms`

Steady-state operator metrics, averaged over iterations 1-2:

| Area | Time | Approx. Wall % | Baseline | Delta |
|---|---:|---:|---:|---:|
| `CuDFLoadExec` elapsed compute | `2275.0 ms` | `53.2%` | `2230.0 ms` | `+45.0 ms` |
| DataSource `time_elapsed_processing` | `1450.0 ms` | `33.9%` | `1390.0 ms` | `+60.0 ms` |
| `CuDFLoadExec input_wait_time` | `1690.0 ms` | `39.5%` | `1635.0 ms` | `+55.0 ms` |
| `CuDFLoadExec cast_time` | `831.6 ms` | `19.4%` | `823.5 ms` | `+8.1 ms` |
| `CuDFLoadExec pin_time` | `565.4 ms` | `13.2%` | `508.4 ms` | `+57.0 ms` |
| `CuDFLoadExec import_time` | `115.9 ms` | `2.7%` | `137.2 ms` | `-21.3 ms` |
| `CuDFLoadExec sync_time` | `8.9 ms` | `0.2%` | `719.3 ms` | `-710.4 ms` |
| `CuDFLoadExec output_send_time` | `283.6 ms` | `6.6%` | `300.1 ms` | `-16.5 ms` |
| `CuDFFilterExec` elapsed compute | `263.7 ms` | `6.2%` | `250.7 ms` | `+13.0 ms` |
| GPU expression projection | `63.3 ms` | `1.5%` | `59.7 ms` | `+3.6 ms` |
| `CuDFAggregateExec` elapsed compute | `592.5 ms` | `13.9%` | `592.3 ms` | `+0.2 ms` |

Nsight Systems warmup plus one measured execution:

- Measured iteration wall time: `4409.6 ms`
- H2D GPU memcpy time: `1.71 s` across warmup+measured, roughly `0.86 s/query`
- GPU kernel time: `1.01 s` across warmup+measured, roughly `0.51 s/query`
- Top kernel: groupby hash aggregation, `562 ms` across warmup+measured, roughly `281 ms/query`
- CUDA API wait is still dominated by `cudaStreamSynchronize`: `2.37 s` across warmup+measured
- Reaper event overhead: `cudaEventCreateWithFlags` was `45.9 ms` across `972`
  events, and `cudaEventQuery` was `3.5 ms` across `1,188` calls

### `cudaStreamSynchronize` Attribution With Reaper

Totals are from the Nsight capture, which includes warmup plus one measured
execution. Divide by roughly two for a per-query number.

| Source | Calls | Time | Approx. per query | Notes |
|---|---:|---:|---:|---|
| First pinned allocation after each upload | `1,856` | `1324.3 ms` | `662.2 ms` | No cuDF NVTX range; this is the first `allocate_sync` after each `from_arrow_host` |
| Remaining pinned buffer `allocate_sync` calls | `22,272` | `132.4 ms` | `66.2 ms` | The rest of the 13 pinned Arrow buffer allocations per input batch |
| cuDF aggregate NVTX range | `1,326` | `587.4 ms` | `293.7 ms` | Groupby/aggregate synchronization |
| cuDF filter/copy_if NVTX ranges | `25,984` | `109.1 ms` | `54.5 ms` | `copy_if` plus nested `thrust::copy_if` |
| cuDF concatenate/string setup NVTX ranges | `1,760` | `87.3 ms` | `43.7 ms` | `concatenate` plus `create_strings_device_views` |
| cuDF binary operation NVTX range | `11,256` | `43.3 ms` | `21.7 ms` | GPU expression work |
| Other / small NVTX or outside active NVTX ranges | `29,898` | `87.8 ms` | `43.9 ms` | Mostly small syncs |

The reaper removes the explicit `CuDFLoadExec.sync_time` fence, but the upload
pipeline immediately returns to pinned-buffer allocation. That path uses
`pinned_mr().allocate_sync(bytes)`, and RMM's generic `allocate_sync` wrapper
synchronizes the default stream before returning. Since the previous batch's
H2D copies are still in flight on that same default stream, the wait reappears
inside the pinned allocation path rather than in the `sync_time` metric.

Net effect in this run: steady wall time did not improve. It moved from
`4190.8 ms` baseline to `4277.3 ms` with the reaper enabled, while the visible
`sync_time` bucket dropped by `710.4 ms`.

## Final Time Movement Summary

The reaper saved time only from the explicit post-upload fence metric:

| Metric / attribution | Baseline | Reaper | Change |
|---|---:|---:|---:|
| `CuDFLoadExec.sync_time` | `719.3 ms/query` | `8.9 ms/query` | `-710.4 ms/query` |
| Nsight explicit post-upload sync attribution | `663.2 ms/query` | effectively `0 ms/query` | about `-663 ms/query` |

That time did not disappear from the query. It moved to the pinned allocation
path used immediately after upload:

| New location with reaper | Time |
|---|---:|
| First `pinned_mr().allocate_sync(bytes)` after each `from_arrow_host` | `662.2 ms/query` |
| Remaining pinned-buffer `allocate_sync` calls | `66.2 ms/query` |
| Reaper event creation/query overhead | about `24.7 ms/query` |

So the exact transfer is: the per-batch `synchronize_default_stream()` call was
removed from `CuDFLoadExec.sync_time`, but the next pinned allocation still
calls RMM `allocate_sync`, which synchronizes the default stream and waits for
the previous batch's H2D copies. The visible metric improved by about
`710 ms/query`, while the same wait reappeared as about `662 ms/query` in
`allocate_sync`, plus small event overhead and run-to-run noise.
