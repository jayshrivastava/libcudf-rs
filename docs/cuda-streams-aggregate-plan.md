# CUDA Streams Aggregate Rollout Plan

This document describes the implementation plan for getting CUDA streams
working for aggregate pipelines in `libcudf-datafusion`.

Target behavior:

```text
GPU segment S, partition p
  -> one CUDA stream
  -> reused across load / aggregate / unload
```

For a query with multiple GPU segments, stream lookup must be keyed by:

```text
(segment_id, partition)
```

not just `partition`.

## Goal

Make these aggregate pipelines stream-aware:

```text
load -> aggregate -> unload
```

and:

```text
load -> aggregate -> unload -> repartition -> load -> aggregate -> unload
```

The second example is two GPU segments separated by a CPU repartition boundary,
so it must use two different stream slots:

```text
(segment 0, partition p) -> stream A
(segment 1, partition p) -> stream B
```

## Stream Ownership Model

The runtime state should be split like this:

```text
CuDFConfig
  -> __private__stream_source

TaskContext
  -> CuDFTaskContext
       -> streams[(segment_id, partition)] = Arc<CuDFStream>
```

Responsibilities:

- `CuDFConfig`
  - owns the private stream allocator / source
  - does not hold per-query mutable stream state
- `CuDFTaskContext`
  - holds per-query stream assignments
  - answers "what stream should this segment+partition use?"
- `CuDFExt`
  - clones a `TaskContext` and installs a fresh `CuDFTaskContext`
  - should be called once per query run, not by individual operators

## Segment ID Model

The optimizer should assign a `segment_id` to each contiguous GPU segment.

Every GPU execution node that participates in stream lookup should carry:

```rust
segment_id: usize
```

For the aggregate rollout, that means at least:

- `CuDFLoadExec`
- `CuDFAggregateExec`
- `CuDFUnloadExec`

Later GPU nodes should carry it too:

- `CuDFFilterExec`
- `CuDFProjectionExec`
- `CuDFSortExec`
- `CuDFHashJoinExec`
- `CuDFCoalesceBatchesExec`

Runtime lookup then becomes:

```text
CuDFTaskContext::stream(segment_id, partition)
CuDFTaskContext::set_stream(segment_id, partition, stream)
CuDFTaskContext::unset_stream(segment_id, partition)
```

## Phase 1: Finish the Explicit-Stream Wrapper Surface

Before wiring execution nodes, make sure the FFI and safe wrappers exist for
every operation the aggregate path uses.

Required `libcudf-sys` entry points:

- `table_from_arrow_host_on`
- `column_from_arrow_on`
- `concat_table_views_on`
- `concat_column_views_on`
- `GroupBy::aggregate_on`
- `TableView::to_arrow_array_on`
- `ColumnView::to_arrow_array_on`
- `cast_column_on`

Required `libcudf-rs` wrappers:

- `CuDFTable::from_arrow_host_on`
- `CuDFColumn::from_arrow_host_on`
- `CuDFTable::concat_on`
- `CuDFColumn::concat_on`
- `CuDFGroupBy::aggregate_on`
- `CuDFTableView::to_arrow_host_on`
- `CuDFColumnView::to_arrow_host_on`
- `cast_on`

This phase is complete when the aggregate pipeline can use explicit streams
without falling back to cuDF's default stream anywhere.

## Phase 2: Make `CuDFTaskContext` Segment-Aware

Refactor the current task context state from:

```text
partition -> stream
```

to:

```text
(segment_id, partition) -> stream
```

Suggested API:

```rust
pub fn stream(&self, segment_id: usize, partition: usize) -> Option<Arc<CuDFStream>>;
pub fn set_stream(&self, segment_id: usize, partition: usize, stream: Arc<CuDFStream>);
pub fn unset_stream(&self, segment_id: usize, partition: usize) -> Option<Arc<CuDFStream>>;
```

This is the minimum needed to make repeated repartitioned aggregate pipelines
behave correctly.

## Phase 3: Annotate GPU Segments in the Optimizer

Update `HostToCuDFRule` so that after CPU nodes are rewritten into cuDF nodes
and `CuDFLoadExec` / `CuDFUnloadExec` are inserted, the resulting plan is
annotated with segment IDs.

For the first rollout:

- every `CuDFLoadExec` starts a new segment
- that `segment_id` propagates upward through the surrounding GPU subplan
- the matching `CuDFAggregateExec` and `CuDFUnloadExec` get the same
  `segment_id`

The result should be:

```text
segment 0:
  CuDFLoadExec(segment_id=0)
  ...
  CuDFAggregateExec(segment_id=0)
  ...
  CuDFUnloadExec(segment_id=0)
```

and after repartition:

```text
segment 1:
  CuDFLoadExec(segment_id=1)
  ...
  CuDFAggregateExec(segment_id=1)
  ...
  CuDFUnloadExec(segment_id=1)
```

## Phase 4: Wire `CuDFLoadExec`

`CuDFLoadExec::execute` becomes the first place that may allocate the stream for
`(segment_id, partition)`.

When `CuDFConfig.cuda_streams == false`:

- keep the current path
- use `CuDFTable::from_arrow_host`

When `CuDFConfig.cuda_streams == true`:

1. get `CuDFTaskContext` from `TaskContext`
2. look up `(segment_id, partition)`
3. if absent, allocate from `CuDFConfig.__private__stream_source`
4. store it in `CuDFTaskContext`
5. use `CuDFTable::from_arrow_host_on`

## Phase 5: Wire `CuDFAggregateExec` and `aggregate/stream.rs`

`CuDFAggregateExec` should not allocate streams.

It should:

1. read the already-assigned stream for `(segment_id, partition)`
2. pass it into `aggregate/stream.rs`

`aggregate/stream.rs` must use that same stream for:

- chunk concat
- groupby aggregate
- running-state merge concat
- any host-array upload in `evaluate_batch_arguments`
- any cast in partial-state output construction

Concretely, the stream-aware path should call:

- `CuDFTable::concat_on`
- `CuDFColumn::concat_on`
- `CuDFGroupBy::aggregate_on`
- `CuDFColumn::from_arrow_host_on`
- `cast_on`

## Phase 6: Wire `CuDFUnloadExec`

`CuDFUnloadExec::execute` should use the same stream that `Load` and aggregate
used for the same `(segment_id, partition)`.

When `cuda_streams == false`:

- keep `to_arrow_host()`

When `cuda_streams == true`:

- use `to_arrow_host_on(stream)`

For the first implementation, `CuDFUnloadExec` can also:

- `unset_stream(segment_id, partition)`

That matches the current intended lifecycle:

```text
load allocates/sets
aggregate reuses
unload reuses and clears
```

If this turns out to be too early for some plans, the fallback is to keep stream
lifetime tied to the query-local `CuDFTaskContext` and drop them at query end.

## Phase 7: Tests

Focused tests to add:

### Wrapper tests

- `libcudf-sys`
  - explicit-stream Arrow upload works
  - explicit-stream concat works
  - explicit-stream groupby aggregate works
  - explicit-stream cast works
- `libcudf-rs`
  - `.on(...)` variants round-trip correctly

### Task context tests

- same `(segment_id, partition)` returns the same stream
- same `partition`, different `segment_id` returns different streams
- `unset_stream` clears only the targeted `(segment_id, partition)`

### Optimizer / plan tests

- aggregate GPU segment gets one `segment_id`
- repartitioned aggregate plan gets two distinct `segment_id`s

### Execution tests

- one-partition aggregate with `cuda_streams = false`
- one-partition aggregate with `cuda_streams = true`
- multi-partition aggregate with `cuda_streams = true`
- repartitioned aggregate pipeline with `cuda_streams = true`

## Phase 8: Benchmarks

Use:

- [libcudf-datafusion/benches/aggregate.rs](/home/ubuntu/libcudf-rs/libcudf-datafusion/benches/aggregate.rs)

Add two GPU benchmark variants:

- `gpu_streams_off`
  - `CuDFConfig.cuda_streams = false`
- `gpu_streams_on`
  - `CuDFConfig.cuda_streams = true`

Important benchmark setup rule:

- create a fresh query-local `TaskContext` for each benchmark iteration
- when streams are enabled, wrap it with `with_cudf_task_context()`

That avoids accidentally reusing mutable stream state across benchmark runs.

Benchmark queries to compare:

- `SUM`
- `COUNT`
- `AVG`
- `MIN/MAX`
- combined aggregate query

## Completion Criteria

This rollout is done when:

- aggregate pipelines use one stream per `(segment_id, partition)` when
  `cuda_streams = true`
- the default-stream behavior is unchanged when `cuda_streams = false`
- repartitioned aggregate pipelines do not reuse the wrong stream slot
- tests pass
- the aggregate benchmark reports separate `gpu_streams_off` vs
  `gpu_streams_on` results
