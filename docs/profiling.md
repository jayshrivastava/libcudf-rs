# Profiling

`libcudf-datafusion` runs work on the GPU across multiple CUDA streams, so
wall-clock benchmarks are not enough to tell whether streams are actually
overlapping or whether something is serializing them. Use Nsight Systems for
that.

## Tooling

`nsys` is NVIDIA's system-wide tracer. It records a timeline of CUDA API
calls, kernel launches, memcpy operations, and NVTX ranges, then opens it in
a GUI where each stream is a separate lane.

Install on the GPU box:

```bash
sudo apt-get install -y nsight-systems-2025.6.3
```

Install the macOS host (viewer only, no GPU needed) from
https://developer.nvidia.com/nsight-systems/get-started — separate Apple
Silicon and Intel downloads, free NVIDIA developer account required. Keep
the macOS host version >= the Linux CLI version that produced the report.

## Recording a trace

```bash
nsys profile -o <name> --trace=cuda,nvtx <command>
```

Example for the aggregate bench:

```bash
nsys profile -o agg_streams_on --trace=cuda,nvtx --force-overwrite=true \
  cargo bench -p libcudf-datafusion --bench aggregate -- \
  combined/gpu_streams_on/20000000 --sample-size 10
```

Notes:

- Tracing adds ~3-5% overhead. Don't read wall-clock numbers from a traced
  run as real perf — use the trace for shape only.
- `--trace=cuda,nvtx` is enough for stream/copy/kernel analysis. Add
  `osrt` for syscalls or `nvtx` ranges if the code emits them.
- The bench picks up `CUDF_BENCH_TARGET_PARTITIONS` /
  `CUDF_BENCH_INPUT_PARTITIONS` from the environment; pass them through
  `nsys profile` like any other env var.

The report file is `<name>.nsys-rep`.

## Viewing the trace

```bash
scp <ubuntu-host>:/path/to/<name>.nsys-rep .
open <name>.nsys-rep
```

## Quick CLI summaries

Useful when you want totals without leaving the terminal:

```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum <name>.nsys-rep
```

Other reports worth knowing:

- `cuda_api_sum` — top CUDA API calls by time (catches `cudaStreamSynchronize` hot spots)
- `cuda_gpu_trace` — per-event trace, expensive for big files

## What to look for

- Do per-partition CUDA streams have overlapping kernel/copy blocks, or do
  they form a staircase? A staircase means streams are serializing.
- Are there gaps where every stream pauses at once? Those are usually
  `cudaStreamSynchronize` calls hiding inside arrow-host transfer or RMM.
- What fraction of wall time is groupby kernels vs. H↔D copy vs. allocator?
- When only one stream has work, are SMs near saturation? If yes, additional
  streams have nowhere to run.
- T4 has only 2 copy engines — H→D copies on different streams may queue on
  the same engine and not overlap even when streams are otherwise concurrent.
