# Why `cudaMallocFromPoolAsync` is 3.5× slower with streams, and how to fix it elegantly

## Recap of the symptom

| Metric                              | streams_on | streams_off | Δ        |
|-------------------------------------|------------|-------------|----------|
| `cudaMallocFromPoolAsync` avg ns    | 35,749     | 10,292      | **3.47×** |
| `cudaMallocFromPoolAsync` median ns | 16,150     | 4,490 (impl) | ~3.6×   |
| `cudaMallocFromPoolAsync` max ns    | 20,994,040 | 16,035,522   | ~1.3×    |
| `cudaStreamWaitEvent` calls         | 989        | 7           | 141×     |
| Total alloc-API time / iter         | ~91 ms     | ~41 ms      | 2.2×     |

305K of the 315K `cudaMallocFromPoolAsync` calls are issued by cuDF's
`make_fixed_width_column` / `make_numeric_column` constructors during
`from_arrow_host` upload — i.e. one device alloc per Arrow input column per
batch.

## What's actually happening

We currently do this in `libcudf-sys/src/operations.cpp`:

```cpp
static std::unique_ptr<rmm::mr::cuda_async_memory_resource> async_mr;
async_mr = std::make_unique<rmm::mr::cuda_async_memory_resource>(initial, max);
rmm::mr::set_current_device_resource(async_mr.get());
```

That's **one** `cuda_async_memory_resource`, which wraps **one** CUDA mempool
(`cudaDeviceGetDefaultMemPool` or one created by RMM). Every cuDF op on every
partition allocates from that single pool, on its own stream.

When a single pool is fed by N concurrent streams, the CUDA driver does
work that doesn't show up at lower N:

1. **Cross-stream reuse ordering.** When the pool wants to satisfy an alloc
   on stream A by recycling a block that was last freed on stream B, the
   driver inserts a stream-wait-event so stream A waits on B's completion
   for that block. We see 989 explicit `cudaStreamWaitEvent` calls in the
   profile (vs 7 in streams_off) — and the *implicit* ones inside
   `cudaMallocFromPoolAsync` are what's eating the per-call cost.
2. **Pool metadata lock.** The pool keeps freelists; concurrent alloc/free
   from 4 worker threads serializes on its internal locks.
3. **Pool growth tail.** When the pool runs short, it does a real
   `cudaMalloc` under the hood. That's where the 21 ms tail per call comes
   from. With more streams, you hit the high-water more often.

The clean version of this fact: **a CUDA mempool is fastest when one stream
owns it.** With one stream the driver elides every cross-stream check; with
many streams it can't.

## The structural detail that makes this fixable

In our pipeline, **one (segment, partition) maps to exactly one stream**.
`CuDFLoadExec`, `CuDFAggregateExec`, `CuDFUnloadExec` for partition i all
look up the same `CuDFStream` from `CuDFTaskContext`. So:

- All allocs for partition i happen on stream i.
- All frees for partition i happen on stream i.
- No partition's data ever crosses to another partition's stream.

This is the precondition that makes per-stream pools both safe and
effective. There's no cross-stream reuse opportunity to lose, because there
is no cross-stream traffic in the first place.

## Options, ranked

### Option 1 (drop-in): per-thread arena on top of the existing pool

```cpp
#include <rmm/mr/device/arena_memory_resource.hpp>

static std::unique_ptr<cuda_async_memory_resource> upstream_mr;
static std::unique_ptr<arena_memory_resource<cuda_async_memory_resource>> arena_mr;

upstream_mr = std::make_unique<cuda_async_memory_resource>(initial, max);
arena_mr    = std::make_unique<arena_memory_resource<cuda_async_memory_resource>>(
                  upstream_mr.get());
rmm::mr::set_current_device_resource(arena_mr.get());
```

`arena_memory_resource` keeps a per-thread arena. Allocations land in the
calling thread's arena (one tokio worker → one arena → one stream in our
model), and only fall back to upstream when the arena needs a new
"superblock". Result: the hot path is a thread-local bump pointer, with
zero contention and zero cross-stream events.

- **Pros:** one-line config change, fully RMM-supported, doesn't change any
  cuDF call sites.
- **Cons:** arenas can fragment under bursty alloc/free patterns; tail
  memory utilization may be ~1.2–1.5× worse than a tight pool.

### Option 2 (surgical): one CUDA mempool per stream

Write a small C++ MR that owns N `cudaMemPool_t` (or N
`cuda_async_memory_resource`) instances and dispatches `do_allocate` by
stream:

```cpp
class per_stream_async_mr : public rmm::mr::device_memory_resource {
    std::unordered_map<cudaStream_t, std::unique_ptr<cuda_async_memory_resource>> pools_;
    std::mutex mtx_;  // only on first-touch per stream

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        cuda_async_memory_resource* mr;
        {
            std::lock_guard lock(mtx_);
            auto& slot = pools_[stream.value()];
            if (!slot) slot = std::make_unique<cuda_async_memory_resource>(...);
            mr = slot.get();
        }
        return mr->allocate(bytes, stream);
    }

    void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream) override {
        // cudaFreeAsync routes by ptr internally; any pool's MR works here.
        // (We pick the pool we allocated from for symmetry / correctness.)
        // ... lookup-or-call cudaFreeAsync directly
    }
};
```

- **Pros:** eliminates the actual root cause (cross-stream coordination in
  a shared pool); per-call cost should approach the streams_off baseline.
- **Cons:** more code; N pools × min-pool-size = more reserved memory,
  though each is fed by the same cudaMalloc backing so it's "soft"
  reservation.

### Option 3 (free, probably insufficient): pool attribute tuning

```cpp
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, device);

// Probably already set by RMM, but worth verifying:
size_t threshold = UINT64_MAX;
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

int enable = 1;
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReuseFollowEventDependencies, &enable);
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReuseAllowOpportunistic, &enable);
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReuseAllowInternalDependencies, &enable);
```

These shave the *price* of cross-stream ordering (the driver may reuse
without inserting an event when it can prove safety). They don't *avoid*
it. Free experiment — useful as a first sanity check — but unlikely to
close the 3.5× gap on its own.

### Option 4 (orthogonal lever): bigger batches

8K rows × 2 columns means ~2 allocs per batch. At 64K rows the alloc count
drops 8× and the per-iter total alloc cost drops with it, *regardless* of
which pool we use. This is the cheapest experiment to confirm the
`make_*_column`/alloc count is the real driver — bench at 64K row batches
and watch `make_fixed_width_column` instances drop 8× and total alloc time
drop with it.

## Recommendation

Try **Option 1 (arena_memory_resource)** first.

- It's a 5-line config change.
- It directly attacks the contention path (per-thread fast path, very few
  upstream calls).
- If it works, we keep streams_on as a clean win on every shape and never
  write a custom MR.

If Option 1 leaves residual per-call cost in the upstream
`cuda_async_memory_resource` (the arena periodically grabs new
superblocks), escalate to **Option 2** for full per-stream isolation. At
that point the upstream cost is paid only at superblock granularity, which
is rare.

In parallel, run **Option 4** (bigger batches) as the cheapest control:
it'll tell us whether streams swing positive once the alloc count is
reduced, even before changing the MR. That confirms the diagnosis.

## Concrete validation plan

1. Bench `sum/streams_on/20M` at default 8K batch size — current baseline.
2. Bench same with `CUDF_BENCH_BATCH_SIZE=65536` (or whatever knob exists)
   — confirms alloc count is the lever.
3. Land Option 1; re-bench 1 and 2.
4. If 1 still slower than streams_off, land Option 2.

Each step is reversible and gated on the previous step's measurement.
