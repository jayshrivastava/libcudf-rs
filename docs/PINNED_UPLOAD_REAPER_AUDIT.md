# Pinned Upload Reaper Audit

## Issue 1: Fallible global initialization

Symptom:
`cargo test -p libcudf-rs pinned` failed with `use of unstable library feature once_cell_try`.

Root cause:
`std::sync::OnceLock::get_or_try_init` is still unstable on the active Rust toolchain.

Fix:
Replaced it with a manual `OnceLock::get` / `try_spawn` / `set` sequence. Failed initialization does not populate the global reaper.

Why this preserves the lifetime model:
If initialization fails after an upload was enqueued, `submit_uploaded_pinned_batch` synchronizes cuDF's default stream before dropping the submitted pinned batch.

Remaining risk:
Concurrent first submissions may spawn a losing short-lived reaper task whose sender is dropped immediately. That task has no submitted batches and exits.

## Issue 2: Submit failure after upload

Symptom:
The planned call order submits the pinned batch after `CuDFTable::from_arrow_host(pinned_batch.clone())` returns; if reaper initialization or channel send failed, the pinned batch could be dropped while copies were still in flight.

Root cause:
The reaper takes ownership after the FFI upload call, so the error path still has to protect the source buffers.

Fix:
On initialization failure or closed-channel send failure, synchronize cuDF's default stream before dropping the pinned batch and returning an error.

Why this preserves the lifetime model:
The fallback synchronization is only on an error path. Successful reaper-enabled uploads still keep pinned host buffers alive with a CUDA event instead of synchronizing per batch.

Remaining risk:
If CUDA stream synchronization itself fails, the process is already in an error state and the pinned batch will still be dropped during unwinding.

## Issue 3: Reaper event errors

Symptom:
Event create/record/query failures inside the background task would otherwise panic and drop open pinned batches immediately.

Root cause:
The sample reaper used `expect(...)` directly around event operations.

Fix:
The reaper synchronizes cuDF's default stream before panicking on seal/query errors.

Why this preserves the lifetime model:
The sync gives outstanding default-stream uploads a chance to complete before the task drops pinned batches.

Remaining risk:
Runtime shutdown can abort the task. Normal DataFusion execution keeps the Tokio runtime active for the query, but short-lived runtimes should not submit uploads and then exit immediately.

## Issue 4: Tokio feature set

Symptom:
The plan requested only Tokio `rt`, `sync`, and `time`, while the sample used `tokio::select!`, which requires the `macros` feature.

Root cause:
`tokio::select!` is gated separately from the requested minimal features.

Fix:
Implemented the task loop with `tokio::time::timeout`, `UnboundedReceiver::recv`, and `try_recv`.

Why this preserves the lifetime model:
Groups are still sealed only after upload calls return, and each sealed group records exactly one CUDA event on cuDF's default stream.

Remaining risk:
The loop is polling based, so group sealing has up to the configured polling interval of extra latency.

## Issue 5: Benchmark binary freshness

Symptom:
The pre-existing `target/release/dfbench` could not load `libcudf.so`.

Root cause:
The binary was stale and lacked the current build's usable `$ORIGIN/deps` runpath.

Fix:
Rebuilt `dfbench` with `cargo build --release -p libcudf-datafusion-benchmarks --bin dfbench`.

Why this preserves the lifetime model:
No runtime behavior changed; it only ensured the benchmark binary included the new reaper code and current linking metadata.

Remaining risk:
None for the reaper implementation.

## Issue 6: Reaper wait moves into pinned allocation

Symptom:
With `LIBCUDF_PINNED_UPLOAD_REAPER=1`, `CuDFLoadExec.sync_time` dropped from
about `719 ms/query` to about `9 ms/query`, but steady TPCH Q1 wall time did
not improve. Nsight still showed about `2.37 s` of `cudaStreamSynchronize`
across warmup plus one measured query.

Root cause:
The explicit post-upload default-stream fence was removed, but the next batch's
pinning path immediately allocates pinned buffers through
`pinned_mr().allocate_sync(bytes)`. RMM's generic `allocate_sync` wrapper calls
`stream.synchronize()` before returning. Because the previous batch's H2D
copies are still queued on the default stream, the wait moves from
`CuDFLoadExec.sync_time` into the next batch's `pin_time`.

Fix:
No code change made in this pass. The benchmark note records the attribution in
`docs/TPCH_Q1_P1_BS65536_BREAKDOWN.md`.

Why this preserves the lifetime model:
The current behavior is conservative: synchronization before the next pinned
allocation prevents early reuse/free and does not shorten the pinned source
lifetime. It only means the reaper does not yet remove the upload wait from the
critical path.

Remaining risk:
To make the reaper effective, pinned host allocation likely needs an async or
non-synchronizing allocation path that is valid for cuDF's pinned memory
resource, while still preserving allocation ownership and default-stream
ordering.
