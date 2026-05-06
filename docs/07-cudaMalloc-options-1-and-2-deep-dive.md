# Options 1 and 2 in detail

Both options change *only* `libcudf-sys/src/operations.cpp`. cuDF reads
`rmm::mr::get_current_device_resource()` to pick where allocations go;
swapping that resource is enough — no cuDF or libcudf-rs code needs to
change.

---

## Today's setup (the baseline we're trying to beat)

```
                    rmm::mr::set_current_device_resource(async_mr)
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────┐
       │            cuda_async_memory_resource                 │
       │  ┌─────────────────────────────────────────────────┐  │
       │  │   ONE CUDA mempool (cudaMemPool_t)              │  │
       │  └─────────────────────────────────────────────────┘  │
       └────────▲──────────────▲──────────────▲───────────────┘
                │              │              │
       ┌────────┴────┐ ┌───────┴─────┐ ┌──────┴──────┐
       │  CuDFStream │ │  CuDFStream │ │  CuDFStream │   ... (one per partition)
       │      0      │ │      1      │ │      2      │
       └─────────────┘ └─────────────┘ └─────────────┘
```

All allocate/free traffic funnels through one pool. The driver inserts
cross-stream ordering on every reuse → 36 µs per call avg, 21 ms tails.

A note on threads: tokio's work-stealing scheduler means a future awaiting on
a CuDFStream may be polled on a different worker thread between batches.
Both options below key allocator state on the `cudaStream_t` handle (not on
`std::this_thread::get_id()`), so this thread mobility is invisible to
either allocator.

---

## Option 1 — `arena_memory_resource` on top of the existing pool

### Picture

We pass non-default streams in, so RMM's arena resource takes the
`stream_arenas_` path (keyed by `cudaStream_t`), not the per-thread path.
The per-thread variant only triggers for the per-thread-default-stream
sentinel, which we never use.

```
                rmm::mr::set_current_device_resource(arena_mr)
                                    │
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │      arena_memory_resource<cuda_async_memory_resource>         │
   │                                                                │
   │   stream_arenas_  =  std::map<cudaStream_t, arena>             │
   │                                                                │
   │     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
   │     │ stream 0  →  │  │ stream 1  →  │  │ stream 2  →  │  ...  │
   │     │ arena 0      │  │ arena 1      │  │ arena 2      │       │
   │     │  superblocks │  │  superblocks │  │  superblocks │       │
   │     │   free_blks_ │  │   free_blks_ │  │   free_blks_ │       │
   │     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
   │            └─────────────────┼─────────────────┘               │
   │                              ▼                                 │
   │                  ┌────────────────────┐                        │
   │                  │   global arena     │  (pulls superblocks    │
   │                  │   (one mutex)      │   from upstream MR)    │
   │                  └─────────┬──────────┘                        │
   └────────────────────────────│───────────────────────────────────┘
                                ▼
       ┌───────────────────────────────────────────────────────┐
       │     cuda_async_memory_resource (the existing pool)    │
       └───────────────────────────────────────────────────────┘

   tokio workers float freely above this; the stream handle they pass in
   is what selects the arena.
```

A **superblock** is a 1 MiB+ contiguous range pulled from the upstream
pool; the arena sub-allocates from it via a `std::set<block>` ordered by
address (`free_blocks_`).

### Hot-path code (RMM, abridged)

`do_allocate` takes a `shared_lock`, finds the stream's arena, walks its
free list:

```cpp
void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
{
    bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto& arena = get_arena(stream);                 // → stream_arenas_[stream]

    {
        std::shared_lock lock(mtx_);
        void* pointer = arena.allocate(bytes);       // ← see first_fit below
        if (pointer != nullptr) { return pointer; }
    }
    // arena exhausted → defragment cliff (see Caveats)
    ...
}
```

Inside the arena, allocation walks `free_blocks_` linearly:

```cpp
block first_fit(std::size_t size)
{
    auto const fits = [size](auto const& blk) { return blk.fits(size); };
    auto const iter = std::find_if(free_blocks_.cbegin(),
                                   free_blocks_.cend(), fits);   // ← O(N)
    if (iter == free_blocks_.cend()) { return {}; }
    if (size < iter->size()) {
        auto const split = iter->split(size);
        free_blocks_.insert(next, split.second);                  // O(log N)
    }
    return result;
}
```

`do_deallocate` tries to return into the same arena and coalesce:

```cpp
void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
{
    auto& arena = get_arena(stream);
    {
        std::shared_lock lock(mtx_);
        if (arena.deallocate(ptr, bytes, stream)) { return; }     // common path
    }
    // ptr came from a different arena (rare for our pipeline; see Caveats)
    stream.synchronize_no_throw();                                // ← per-call sync!
    std::unique_lock lock(mtx_);
    deallocate_from_other_arena(ptr, bytes, stream);
}
```

Where `arena.deallocate` does an O(log N) `set::lower_bound` and a
constant-cost neighbour-check to coalesce adjacent free blocks.

### Cost characterisation

When `free_blocks_` is small and stable (uniform-size LIFO alloc/free —
exactly our SUM-on-numeric-cols pattern), N stays at 1–3 and the linear
walk is dozens of ns. When sizes mix and frees don't coalesce (e.g.
variable-width strings, dynamic groupby keys), N can grow to thousands
and per-call cost rises into single µs.

### Code (libcudf-sys/src/operations.cpp)

```cpp
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

bool config_device_memory_pool(size_t initial_bytes, size_t max_bytes) {
    using upstream_t = rmm::mr::cuda_async_memory_resource;
    using arena_t    = rmm::mr::arena_memory_resource<upstream_t>;

    static std::unique_ptr<upstream_t> upstream_mr;
    static std::unique_ptr<arena_t>    arena_mr;
    if (arena_mr) return false;

    upstream_mr = std::make_unique<upstream_t>(
        std::optional<size_t>{initial_bytes},
        std::optional<size_t>{max_bytes});

    // arena_size = total budget for the global arena.
    // If unset, defaults to half of available device memory.
    arena_mr = std::make_unique<arena_t>(upstream_mr.get());

    rmm::mr::set_current_device_resource(arena_mr.get());
    return true;
}
```

That's the whole change.

### Caveats

- **`first_fit` is O(N) per allocation**, where N is the count of free
  blocks in the stream's arena. Stays small for uniform alloc/free
  patterns; grows under fragmentation.
- **`defragment()` calls `cudaDeviceSynchronize`.** It's invoked when the
  arena fails to satisfy a request and has to reclaim across arenas:

  ```cpp
  void defragment()
  {
      RMM_CUDA_TRY(cudaDeviceSynchronize());            // ← FULL DEVICE STALL
      for (auto& thread_arena : thread_arenas_) thread_arena.second->clean();
      for (auto& stream_arena : stream_arenas_) stream_arena.second.clean();
  }
  ```

  A full device sync stalls every stream until everything submitted is
  done — 100+ ms latency cliff. Fires only on arena exhaustion, but the
  cliff exists.
- **Cross-arena deallocate has `stream.synchronize_no_throw()`**. Hits
  when ptr came from a different stream's arena (e.g. cuDF doing internal
  stream juggling). For our load → aggregate → unload pipeline all on one
  stream this shouldn't fire.
- **Memory overhead from arena fragmentation** typically 1.2–1.5× a tight
  pool, depending on size mix.
- **Battle-tested.** This is RAPIDS' default for many production
  workloads. The footguns are real but well-understood.

---

## Option 2 — one CUDA mempool per stream

### Picture

```
                 rmm::mr::set_current_device_resource(per_stream_mr)
                                    │
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │             per_stream_async_memory_resource                   │
   │                                                                │
   │   pools_  =  unordered_map<cudaStream_t,                       │
   │                            unique_ptr<cuda_async_memory_resource>>
   │                                                                │
   │     ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
   │     │ stream 0  →     │  │ stream 1  →     │  │ stream 2  →  │ │
   │     │ cuda_async_mr#0 │  │ cuda_async_mr#1 │  │ cuda_async_mr│ │
   │     │   pool#0        │  │   pool#1        │  │   pool#2     │ │
   │     └─────────────────┘  └─────────────────┘  └──────────────┘ │
   └────────▲──────────────────▲──────────────────▲────────────────┘
            │                  │                  │
       CuDFStream         CuDFStream         CuDFStream
            0                  1                  2

       (tokio workers float freely; only the stream handle picks the pool)
```

Each stream has *its own CUDA mempool*. The driver only ever sees one
stream per pool, so:

- cross-stream reuse ordering: not applicable — there's only one stream
- pool metadata lock: uncontested
- pool growth: per-stream, smaller per-pool, less likely to coincide

### What happens on each path

**`do_allocate(bytes, stream)`:**

```
1. Look up pools_[stream].
   - shared-lock if found (read-mostly)
   - unique-lock + insert on first-touch only
2. Forward to that resource's allocate(bytes, stream)
   → cudaMallocFromPoolAsync(ptr, bytes, pool[stream], stream)
3. Driver returns immediately; pool is hot, single-stream, no events.
```

**`do_deallocate(ptr, bytes, stream)`:**

```
1. Forward to any pool's resource → cudaFreeAsync(ptr, stream)
   The driver knows from `ptr` which pool owns it; no lookup needed
   on our side. We still go through resource_for(stream) so RMM's
   accounting stays consistent.
```

### Code (libcudf-sys/src/per_stream_mr.cpp — new file)

```cpp
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>
#include <shared_mutex>

class per_stream_async_memory_resource final
    : public rmm::mr::device_memory_resource {
public:
    per_stream_async_memory_resource(std::optional<size_t> initial,
                                     std::optional<size_t> max)
        : initial_{initial}, max_{max} {}

    bool supports_streams() const noexcept override { return true; }
    bool supports_get_mem_info() const noexcept override { return false; }

private:
    void* do_allocate(std::size_t bytes,
                      rmm::cuda_stream_view stream) override {
        return resource_for(stream)->allocate(bytes, stream);
    }

    void do_deallocate(void* p, std::size_t bytes,
                       rmm::cuda_stream_view stream) override {
        // cudaFreeAsync routes by `p` internally — pool agnostic.
        resource_for(stream)->deallocate(p, bytes, stream);
    }

    rmm::mr::cuda_async_memory_resource* resource_for(
        rmm::cuda_stream_view stream)
    {
        cudaStream_t key = stream.value();
        {
            std::shared_lock lock(mtx_);
            auto it = pools_.find(key);
            if (it != pools_.end()) return it->second.get();
        }
        std::unique_lock lock(mtx_);
        auto& slot = pools_[key];
        if (!slot) {
            slot = std::make_unique<rmm::mr::cuda_async_memory_resource>(
                initial_, max_);
        }
        return slot.get();
    }

    std::optional<size_t> initial_;
    std::optional<size_t> max_;
    std::shared_mutex mtx_;
    std::unordered_map<cudaStream_t,
                       std::unique_ptr<rmm::mr::cuda_async_memory_resource>>
        pools_;
};
```

And in `operations.cpp`:

```cpp
#include "per_stream_mr.h"

bool config_device_memory_pool(size_t initial_bytes, size_t max_bytes) {
    static std::unique_ptr<per_stream_async_memory_resource> mr;
    if (mr) return false;
    mr = std::make_unique<per_stream_async_memory_resource>(
        std::optional<size_t>{initial_bytes},
        std::optional<size_t>{max_bytes});
    rmm::mr::set_current_device_resource(mr.get());
    return true;
}
```

### How allocations look in steady state

```
   stream 0      pool#0
   ─ alloc ────► hot path, single stream, ~3–5 µs (driver-side only)
   ─ free  ────► returns to pool#0; available immediately on next stream-0 alloc

   stream 1      pool#1
   ─ alloc ────► same, independent of stream 0
```

No cross-stream events because each pool only ever sees one stream. No
software allocator on the hot path; the driver's stream-ordered pool
implementation does the work, but uncontended.

### Caveats

- **Memory ceiling.** Each pool reserves its own working set. With 4
  streams and a 50 MB working set per partition, that's up to 200 MB
  cached vs ~50–80 MB with one shared pool. `cuda_async_memory_resource`
  is lazy though — it commits memory under
  `cudaMemPoolAttrReleaseThreshold`, so this is the steady-state ceiling,
  not 4× upfront.
- **First-touch lock per stream.** Once per (stream, this-resource) pair.
  With one stream per partition this is ~4 lookups in a long-running
  bench.
- **Stream lifecycle.** When a `CuDFStream` is destroyed, its pool sticks
  around in `pools_`. That's a small leak per never-reused stream. Easy
  fix: hook a `release(stream)` from `CuDFStream::Drop`. Not required for
  correctness — just memory cleanliness over very long runs.
- **`supports_streams = true` is required.** Some upstream MRs (e.g.
  `cuda_memory_resource`) ignore the stream argument and would defeat the
  design. We pass through to `cuda_async_memory_resource`, which respects
  it.
- **No software-allocator footgun.** No linear free-list walk, no
  `cudaDeviceSynchronize` cliff. Only failure mode is `cudaErrorMemoryAllocation`
  if the pool's max is set too low.

---

## Side-by-side

|                                  | Option 1 (arena)                              | Option 2 (per-stream pools)                        |
|----------------------------------|-----------------------------------------------|----------------------------------------------------|
| Lines of code                    | ~5 in `operations.cpp`                        | ~50 + new file                                     |
| Hot path                         | `shared_lock` + O(N) walk on `free_blocks_`   | `cudaMallocFromPoolAsync` on uncontended pool      |
| Hot-path cost (uniform sizes)    | tens of ns                                    | ~3–5 µs (driver path, but no cross-stream events)  |
| Hot-path cost (mixed sizes)      | up to single µs (depends on N)                | unchanged — driver's freelist isn't size-sensitive |
| Cold path                        | superblock pull from upstream                 | pool growth (`cudaMalloc` once per pool)           |
| Cliff                            | `defragment()` → `cudaDeviceSynchronize` (100+ ms) on arena exhaustion | `cudaErrorMemoryAllocation` if max too low          |
| Cross-stream coordination cost   | eliminated (per-stream arena)                 | eliminated (per-stream pool)                       |
| Tokio thread mobility            | irrelevant — keyed by `cudaStream_t`          | irrelevant — keyed by `cudaStream_t`               |
| Memory overhead                  | 1.2–1.5× from fragmentation                   | up to N× steady-state working set (lazy commit)    |
| RMM-supported out of the box     | yes                                           | no — small custom MR                               |
| Risk surface                     | software allocator + sync cliff               | custom code, edge cases on stream destroy          |
| Worst-case behavior              | fragmentation → growing N → growing per-call cost | works the same regardless of size mix          |

## Recommendation

**Go straight to Option 2.**

The earlier inclination toward Option 1 was based on "5 lines of code, drop
in" being the elegant move. Inspecting the implementation flips that:

1. **The hot path is genuinely software-allocator-shaped** — `std::set`
   walk + tree insert. For uniform-size workloads it's fine; for mixed
   sizes it has a real worst case.
2. **`defragment()`'s `cudaDeviceSynchronize` is a latency cliff** that
   we'd be one-bad-shape away from hitting under load. Streams are
   supposed to give us latency *predictability*; a 100 ms stall hidden
   inside the allocator works against that goal.
3. **Option 2's "downside" is mostly mechanical** — a small custom MR and
   memory-cleanup hook on stream drop. There are no behavioural cliffs.
   The driver's pool implementation is highly-tuned C++ inside CUDA and
   stays uncontended in our model.
4. **Memory ceiling is bounded and tunable.** N pools × per-pool max is
   predictable; we can size each pool's max small if needed since each
   only services one stream's working set.

The only argument for Option 1 first is "less code." But ~50 lines of
straightforward C++ that wraps an existing RMM resource is not a big lift,
and the resulting allocator has no surprises.

If we ever wanted both, they compose:
`arena_memory_resource<per_stream_async_memory_resource>`. But there's no
reason to start with the arena — we'd be paying its risk to amortize
something Option 2 already amortizes (pool growth) at minimal cost.

### Validation plan

1. Land Option 2 behind the existing `config_device_memory_pool` entry
   point (no API change).
2. Re-run the streams_on / streams_off bench at 20 M for all five shapes.
   Expected: `cudaMallocFromPoolAsync` per-call avg drops from 36 µs back
   toward the streams_off baseline (~10 µs); streams_on becomes a clean
   win across the matrix.
3. Validate with the existing nsys profile recipe — kernel concurrency
   should now translate to wall-time wins now that the alloc side is no
   longer the bottleneck.
4. Add a `release(stream)` hook from `CuDFStream::Drop` to clean up the
   `pools_` entry on stream destruction (memory hygiene, not correctness).

If after step 2 there's still per-call cost we want to shave (i.e. each
pool's individual `cudaMallocFromPoolAsync` is still doing meaningful
driver work), revisit Option 1 as a *layer on top of* Option 2 — at that
point the arena's superblock-granularity calls would land on per-stream
pools, eliminating both contention sources.
