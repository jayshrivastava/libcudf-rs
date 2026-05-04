# RFC: Per-Partition CUDA Streams

## 1. What a CUDA Stream Is

A CUDA stream is an ordered queue of GPU work.

- operations submitted to the same stream run in issue order
- operations submitted to different streams may overlap if the GPU has
  available resources

```text
stream 0:  load -> filter -> aggregate -> unload
stream 1:  load -> join   -> sort      -> unload
```

This does **not** mean both streams always run at the same time. It means the
CUDA runtime is allowed to schedule them concurrently.

For this project, a stream is just a way to say:

```text
"these GPU operations belong to one execution lane"
```

## 2. One Stream Per DataFusion Partition

For a query with `target_partitions = N`, the proposal is to create `N` CUDA
streams and assign one stream to each partition.

```text
query
  partition 0 -> stream[0]
  partition 1 -> stream[1]
  partition 2 -> stream[2]
  ...
  partition N-1 -> stream[N-1]
```

Within one partition, GPU work stays on that partition's stream:

```text
partition p on stream[p]:
  CuDFLoadExec
    -> filter
    -> projection
    -> aggregate
    -> sort
    -> CuDFUnloadExec
```

That gives us:

- ordering within a partition without extra synchronization
- possible overlap across partitions

```text
stream[0]: load -> filter -> aggregate -> unload
stream[1]: load -> join   -> sort      -> unload
stream[2]: load -> filter -> sort      -> unload
```

Today, CPU-only operators like `RepartitionExec` already break the GPU pipeline:

```text
GPU subplan
  -> unload to CPU
  -> repartition on CPU
  -> load back to GPU
```

So repartition does not currently imply GPU buffer sharing across streams.

## 3. `CollectLeft` Join Caveat

Most operators are naturally partition-local under this model. The main
exception is `CuDFHashJoinExec` in `PartitionMode::CollectLeft`.

In that mode:

```text
left child runs once
  -> shared_left : Arc<CuDFTable>

partition 0: join(shared_left, right_0) on stream[0]
partition 1: join(shared_left, right_1) on stream[1]
partition 2: join(shared_left, right_2) on stream[2]
```

So the same GPU-resident left table may be read by multiple streams.

The important distinction is:

```text
safe-ish case:
  stream[0] reads shared_left
  stream[1] reads shared_left

dangerous case:
  stream[0] writes shared_left
  stream[1] reads or writes shared_left
```

The current join implementation appears to build the shared left table once and
then treat it as read-only input. That means the immediate risk is not a
concurrent read/write race, but we still need to be careful:

- no stream should mutate shared build-side state while another stream reads it
- teardown must not free/reuse shared GPU memory while kernels on another stream
  are still reading it

So `CollectLeft` is the one place where cross-stream GPU memory sharing is
expected and must be reviewed carefully.

## 4. Resource Contention and Error Handling

Multiple streams do not guarantee progress. They compete for the same GPU:

- device memory
- temporary workspace
- SMs / compute capacity
- memory bandwidth
- copy engines / PCIe / NVLink bandwidth

```text
stream[0] wants scratch for aggregate
stream[1] wants scratch for join
stream[2] wants scratch for sort
```

If an operation cannot get the memory it needs, the expected behavior should be:

```text
operation allocation fails
  -> cuDF/RMM returns an error
  -> current operation fails
  -> query fails unless the caller handles it
```

The stream itself should still be considered alive. The failure is at the
operation level, not "the whole stream is poisoned forever".

This RFC assumes:

- resource contention may reduce or eliminate overlap
- OOM or allocator failure is a real runtime error
- operators should propagate those errors cleanly back to DataFusion

## Summary

```text
one query
  -> one stream per partition
  -> ordered execution within each partition
  -> possible overlap across partitions
```

The main correctness caveat is shared GPU state in `CollectLeft` joins.
Everything else is mostly partition-local under the current architecture.

The main runtime caveat is resource contention: streams are cheap, but GPU
memory and compute are not.
