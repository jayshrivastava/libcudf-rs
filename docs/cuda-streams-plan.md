# CUDA Streams Implementation Plan

This document tracks the implementation plan for per-partition CUDA streams in
`libcudf-datafusion`.

For background on what CUDA streams are, why they help, and what currently runs
on GPU, see [cuda-streams.md](/home/ubuntu/libcudf-rs/docs/cuda-streams.md).

## Phase 1: Finish the Stream-Aware Primitive APIs

Goal: make the basic cuDF wrappers accept an explicit stream end to end.

Already started:

- `CuDFStream` wrapper exists
- `.on(...)` variants exist for key operations like load, concat, aggregate, and unload

Finish and verify:

- `libcudf-sys`
  - `table_from_arrow_host_on`
  - `concat_table_views_on`
  - `concat_column_views_on`
  - `GroupBy::aggregate_on`
  - `TableView::to_arrow_array_on`
- `libcudf-rs`
  - `CuDFTable::from_arrow_host_on`
  - `CuDFTable::concat_on`
  - `CuDFGroupBy::aggregate_on`
  - `CuDFTableView::to_arrow_host_on`

Success criterion:

```text
CPU batch -> load_on(stream) -> aggregate_on(stream) -> unload_on(stream)
```

works for one partition without changing planner/runtime policy.

## Phase 2: Add TaskContext-Accessible Stream State

Goal: make the current partition stream easy to retrieve from `execute(partition, context)`.

The simplest model is:

- store query-scoped GPU state as a typed `SessionConfig` extension
- expose it through a small `CuDFExt` helper/extension trait over `TaskContext`
- keep a private stream source/manager behind that extension
- lazily create one stream slot per partition

Conceptually:

```text
TaskContext
  -> SessionConfig extension
       CuDFExt
         -> __private__stream_source
         -> partition_streams[partition] -> OnceLock<Arc<CuDFStream>>
```

This matches the existing DataFusion pattern of retrieving execution-scoped state
through `context.session_config().get_extension::<T>()`. The operator story becomes:

```text
execute(partition, context)
  -> context.cudf().stream_for_partition(partition)
```

Recommended API shape:

```text
CuDFExt
  -> private stream source / allocator
  -> stream(partition: usize) -> Arc<CuDFStream>
```

Properties of this model:

- lookup is always keyed by `partition`
- `CuDFLoadExec` is often the first caller of `stream(partition)`
- downstream GPU operators call `stream(partition)` again and get the same stream
- no mutable "current stream" slot is stored in shared context

Avoid this model:

```text
LoadExec sets current stream
GPU operators read current stream
UnloadExec unsets current stream
```

because multiple partitions may execute concurrently against the same query context.
Stream lookup must stay partition-keyed, not ambient and mutable.

Properties:

- query-scoped
- one lazy stream slot per partition
- stream created only if that partition reaches a GPU operator

Important:

- do not attach long-lived stream ownership to `SessionContext`
- keep the stream lookup reachable from `TaskContext`, because that is what operators already receive
- the extension object itself is query-scoped runtime state carried through `SessionConfig`,
  not a global session-wide stream pool
- `CuDFUnloadExec` should not own teardown by default; stream lifetime should be owned by the
  query-scoped extension object and released when that state drops

## Phase 3: Thread the Partition Stream Through the Aggregate Pipeline

Goal: prove the model on one complete GPU path.

Wire partition stream lookup into:

- `CuDFLoadExec::execute`
- `CuDFAggregateExec::execute`
- `aggregate/stream.rs`
- `CuDFUnloadExec::execute`

Required `_on(...)` wrappers for this phase:

- load path
  - `CuDFTable::from_arrow_host_on`
- aggregate path
  - `CuDFTable::concat_on`
  - `CuDFGroupBy::aggregate_on`
- unload path
  - `CuDFTableView::to_arrow_host_on`

If any aggregate argument path still converts host-side arrays into cuDF columns directly, it must
also use the explicit-stream wrapper rather than falling back to the default stream.

The aggregate pipeline should use the same partition stream for:

- host -> device load
- chunk concat
- groupby aggregate
- running-state merge concat
- final unload

Target shape:

```text
partition p
  -> context.cudf().stream_for_partition(p)
  -> load_on(S)
  -> concat_on(S)
  -> aggregate_on(S)
  -> unload_on(S)
```

Do not create the stream inside aggregate itself. Aggregate should receive the partition stream
from execution context.

`CuDFLoadExec` will often be the first operator to call `stream(partition)`, which will lazily
allocate the stream if needed. `CuDFUnloadExec` is often the last user, but it should not unset
or return the stream directly in the first implementation.

Tests to add in this phase:

- unit test for the `TaskContext` extension/helper
  - partition `p` returns the same stream handle across repeated lookups
- one-partition aggregate path test
  - `load_on -> aggregate_on -> unload_on`
  - verifies correctness of results
- multi-partition aggregate path test
  - two partitions each get a distinct stream slot
  - verifies correctness of results
- plan/execution test
  - within one GPU segment, `Load`, aggregate, and `Unload` all use the same partition stream

These tests should stay focused on stream propagation and correctness, not on proving actual GPU
parallel speedup.

## Phase 4: Extend to the Other GPU Operators

Once aggregation works, thread the same partition stream into the rest of the
GPU operator surface.

Required wrapper families to add in this phase:

- filter
  - `libcudf-rs`
    - `apply_boolean_mask_on`
  - `libcudf-sys`
    - `apply_boolean_mask(..., stream)`
- projection / expression evaluation
  - `libcudf-rs`
    - `cast_on`
    - `cudf_binary_op_on`
  - `libcudf-sys`
    - `cast(..., stream)`
    - `binary_operation_col_col(..., stream)`
    - `binary_operation_col_scalar(..., stream)`
    - `binary_operation_scalar_col(..., stream)`
- sort / top-k
  - `libcudf-rs`
    - `sort_on`
    - `sort_by_all_on`
    - `stable_sorted_order_on`
    - `gather_on`
  - `libcudf-sys`
    - `sort_table_on`
    - `stable_sort_table_on`
    - `sorted_order_on`
    - `stable_sorted_order_on`
    - `sort_by_key_on`
    - `stable_sort_by_key_on`
    - `gather(..., stream)`
- hash join
  - `libcudf-rs`
    - `inner_join_on`
    - `left_join_on`
    - `full_join_on`
    - `left_semi_join_on`
    - `left_anti_join_on`
  - `libcudf-sys`
    - `inner_join_gather_on`
    - `left_join_gather_on`
    - `full_join_gather_on`
    - `left_semi_join_gather_on`
    - `left_anti_join_gather_on`
- coalesce / remaining helpers used by the above paths
  - ensure stream-aware variants exist anywhere a GPU operator currently falls
    back to the default stream for concat, gather, cast, or expression kernels

At that point a full partition-local GPU segment can stay on one stream:

```text
load -> filter -> projection -> sort -> join -> aggregate -> unload
```

Tests to add in this phase:

- filter test that uses `apply_boolean_mask_on` through `CuDFFilterExec`
- projection test covering:
  - binary expression evaluation on an explicit stream
  - cast on an explicit stream
- sort test covering:
  - full sort on an explicit stream
  - top-k path using `stable_sorted_order_on` and `gather_on`
- hash join test for each supported join family on an explicit stream
- plan-level test that a mixed GPU segment like
  `load -> filter -> projection -> sort -> aggregate -> unload`
  reuses the same partition stream throughout

## Phase 5: Add Global GPU Concurrency Control

Goal: avoid too many active GPU partitions across many queries.

Add a global cap on active GPU partition executions.

This is more important than pooling stream objects.

The limiter should:

- bound the number of concurrently active GPU partition pipelines
- avoid unbounded memory/workspace pressure
- allow lazy stream creation only for admitted partitions

Conceptually:

```text
query A partition 0  acquires GPU slot
query B partition 1  acquires GPU slot
query C partition 2  waits
```

## Phase 6: Handle Cross-Stream Shared-State Cases

Goal: make shared GPU objects safe when reused across partition streams.

Main current case:

- `CuDFHashJoinExec` in `CollectLeft` mode

Important clarification:

- `CollectLeft` does not create a read-while-build race
- the left side is fully materialized once via `OnceCell` before any partition uses it
- the join path reads that shared left table; it does not mutate it

The real concern is narrower:

- one GPU-resident build-side table may be consumed by multiple partition streams
- this is a shared-input / lifetime problem, not a build/probe concurrency problem
- if joins become stream-aware, ensure the shared left table stays valid until all consumer
  streams are done with it

Before enabling stream-per-partition there, decide one of:

- keep `CollectLeft` on one stream initially
- add explicit event-based cross-stream synchronization
- delay stream-aware `CollectLeft` until partitioned joins are done

## Phase 7: Add Focused Tests

Add only a few targeted tests:

- stream wrapper construction and validity
- `TaskContext` stream lookup / reuse behavior
- one-partition aggregate path using `.on(...)`
- multi-partition aggregate path using distinct streams
- plan-level test that one GPU segment reuses a partition stream across load/aggregate/unload

Avoid broad concurrency tests until the runtime/context design is stable.

## Recommended Order

Build in this order:

1. Finish `.on(...)` wrapper surface.
2. Add `TaskContext`-accessible query-scoped stream state.
3. Wire aggregate path end to end.
4. Extend to other GPU operators.
5. Add global GPU concurrency limit.
6. Handle `CollectLeft` and other shared-state edge cases.
