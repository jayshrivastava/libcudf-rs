Streams DO unlock GPU parallelism. CPU/scheduler overhead eats the win.

GPU side — what streams did fix

┌─────────────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┐
│                                 │         streams_off          │                 streams_on                 │
├─────────────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
│ Distinct CUDA streams           │                  1 (default) │ 296 (4 partitions × 2 segments × ~37 iter) │
├─────────────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
│ Total GPU time across all       │      3272 ms / 55 iter = 60  │  4 streams × ~21 ms each = ~21-30 ms/iter  │
│ streams                         │                      ms/iter │                                       wall │
├─────────────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
│ HtoD memcpy total               │                      2265 ms │            2277 ms (essentially identical) │
├─────────────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
│ HtoD avg per copy               │             12 µs (≈83 GB/s) │                               12 µs (same) │
└─────────────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┘

The top 6 streams in streams_on have ~21 ms of GPU work each, all roughly equal — that's "4 partitions running
concurrently" working as designed. Per-iteration GPU wall is ~20-30 ms in streams_on vs ~60 ms in streams_off. So
the GPU side got roughly 2× faster — the parallelism is real.

CPU side — what streams paid for

┌──────────────────────────────┬───────────────────┬───────────────────┐
│                              │    streams_off    │    streams_on     │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ Wall clock (combined/20M)    │            124 ms │            137 ms │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ GPU contribution             │            ~60 ms │            ~25 ms │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ CPU/scheduler not overlapped │            ~64 ms │           ~112 ms │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ cudaMemcpyAsync API time     │             6.4 s │             5.0 s │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ cudaMemcpyAsync calls        │             186 K │             194 K │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ cudaStreamSynchronize        │ 2.7 s, 94 K calls │ 2.7 s, 96 K calls │
├──────────────────────────────┼───────────────────┼───────────────────┤
│ cudaMallocFromPoolAsync      │       188 K calls │       195 K calls │
└──────────────────────────────┴───────────────────┴───────────────────┘

So streams_on shaved 35 ms off the GPU side but added 48 ms to the CPU side. Where the new CPU cost goes:

- 4× per-partition fixed costs. Each partition runs its own tokio task, its own pin_record_batch loop, its own
per-batch synchronize_default_stream (or per-stream sync), its own RMM alloc/free pairs. With each partition doing
1/4 the data, the per-partition fixed cost doesn't shrink linearly. ~7K extra cudaMemcpyAsync and ~7K extra
alloc/frees vs streams_off, despite identical total bytes.
- More tokio worker overhead. 4 workers waking up, polling, re-scheduling per batch — instead of one worker pulling
sequentially.
- Streams-on doesn't reuse pin pool across partitions. The thread-local pool means each tokio worker thread builds
its own pin pool from scratch. With 4 workers, that's 4 cold-start allocations of ~1.3MB pinned buffers per
iteration before the pool warms up.

                               streams_off    streams_on
cub::for_each::static_kernel       536 ms        555 ms     (185 vs 296 instances)
cub::for_each::static_kernel       103 ms        110 ms
fused_concatenate_kernel<double>    53 ms         53 ms
fused_concatenate_kernel<long>      53 ms         53 ms
DeviceSelectSweep                   27 ms         29 ms

Same kernels in roughly equal totals. The 296 vs 185 instance count for the top kernel is what 4-stream parallelism
looks like — more, smaller kernel launches per stream, but each less wall time individually.

So where's the bottleneck on streams_on?

CPU work that runs serially across partitions. Specifically:

1. The per-batch stream.synchronize() after from_arrow_host_on in CuDFLoadExec — one per batch per partition, ~244
batches × 4 partitions = ~1000 explicit syncs per iter. Each is short (<5 µs typically, since GPU is keeping up)
but they're on the hot path of every partition.
2. The cudaMallocFromPoolAsync / cudaFreeAsync chatter — 195 K of each. The driver's async pool isn't perfectly
free; each call takes 11-12 µs avg on the CPU.
3. pin_record_batch runs on whatever tokio worker. Each batch does an arrow→pinned memcpy. With 4 workers all doing
this at ~1.3 MB/batch, you're contending for memory bandwidth on the host side too.

Concrete next levers

In rough order of effort × payoff:

1. Larger batches per partition (raise target_batch_size so each partition processes ~5M rows in fewer batches).
This amortizes per-batch fixed cost — alloc, sync, kernel launch — over more bytes. Cheap, and would directly hit
the bottleneck.
2. Drop the per-batch sync (use the lifetime-guard pattern we discussed earlier — keep pinned batch alive past
from_arrow_host so the implicit cuDF sync covers us). Saves ~244 × 4 syncs/iter on streams_on.
3. Per-partition pinned pool — skip the thread-local model when streams are on, give each partition its own pool
initialized to expected size. Eliminates cold-start malloc per iteration.
4. Skip CuDFAggregateExec's separate sync — currently both LoadExec and AggregateExec end up syncing on the same
stream during normal flow. Could be merged.

The first two would land you at roughly streams_on ≈ streams_off / N (where N = some factor depending on partition
count). The latter two are smaller wins on top.

Want me to try the per-batch sync removal first? It's the easiest knob and we already had the conversation about
how to do it.
