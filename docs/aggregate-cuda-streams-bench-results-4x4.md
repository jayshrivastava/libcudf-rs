# Aggregate CUDA Streams Bench Results - 4 Input Partitions

Date: 2026-05-03

## Run Details

- GPU: Tesla T4, 15360 MiB
- Benchmark: `libcudf-datafusion/benches/aggregate.rs`
- Change under test: `MemTable` input batches are distributed round-robin across input partitions instead of being registered as one input partition.
- Target partitions: 4
- Input partitions: 4
- Command:

```bash
CUDF_BENCH_TARGET_PARTITIONS=4 \
CUDF_BENCH_INPUT_PARTITIONS=4 \
cargo bench -p libcudf-datafusion --bench aggregate -- gpu_streams \
  --sample-size 10 --warm-up-time 1 --measurement-time 5
```

Criterion's `change` output compares each benchmark to previously saved baselines. The
tables below compare streams off vs. streams on from this same run. Raw rows use
Criterion's `slope` estimate, which is the estimate printed in the benchmark output.

## Summary

| Benchmark | Rows | Streams off | Streams on | Speedup |
| --- | ---: | ---: | ---: | ---: |
| sum | 1M | 13.534 ms | 11.480 ms | 1.18x |
| sum | 5M | 40.190 ms | 39.074 ms | 1.03x |
| sum | 20M | 118.166 ms | 107.326 ms | 1.10x |
| count | 1M | 13.436 ms | 13.549 ms | 0.99x |
| count | 5M | 37.757 ms | 37.130 ms | 1.02x |
| count | 20M | 116.139 ms | 112.786 ms | 1.03x |
| avg | 1M | 16.583 ms | 21.219 ms | 0.78x |
| avg | 5M | 43.682 ms | 42.338 ms | 1.03x |
| avg | 20M | 127.545 ms | 119.689 ms | 1.07x |
| min_max | 1M | 15.957 ms | 28.185 ms | 0.57x |
| min_max | 5M | 44.156 ms | 42.275 ms | 1.04x |
| min_max | 20M | 124.593 ms | 121.104 ms | 1.03x |
| combined | 1M | 31.573 ms | 48.088 ms | 0.66x |
| combined | 5M | 69.352 ms | 65.914 ms | 1.05x |
| combined | 20M | 151.422 ms | 141.199 ms | 1.07x |

## Interpretation

The MemTable fix matters. Before this change, `target_partitions=4` did not mean the
source had four input partitions; the benchmark registered one partition containing
all generated batches. With four real input partitions, the streams-off baseline also
improves because DataFusion can run independent input work concurrently.

Streams are still not a dramatic win. For 5M and 20M rows they are usually modestly
faster, roughly 1.02x to 1.10x in this run. That suggests stream concurrency is
overlapping some per-partition host-to-device, aggregate, and device-to-host work, but
the end-to-end query still has shared costs that limit scaling.

The 1M cases are dominated by overhead and variance. `sum` improves, `count` is
effectively neutral, and `avg`, `min_max`, and `combined` regress with wide confidence
intervals. Those small-row results should not be read as steady-state GPU throughput.

The likely remaining limit is not just scheduling. cuDF groupby/reduction kernels,
partial-state merge work, memory allocation/synchronization, and final output movement
still impose serialized or contention-heavy stages. CUDA streams can only overlap work
that is both independent and submitted on different streams.

## Raw Criterion Slope Estimates

Throughput is computed from the point estimate as rows per second in millions.

| Benchmark | Rows | Mode | Time low | Time point | Time high | Throughput (Melem/s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sum | 1M | gpu_streams_off | 12.861 ms | 13.534 ms | 14.418 ms | 73.89 |
| sum | 1M | gpu_streams_on | 11.198 ms | 11.480 ms | 11.658 ms | 87.11 |
| sum | 5M | gpu_streams_off | 39.039 ms | 40.190 ms | 40.910 ms | 124.41 |
| sum | 5M | gpu_streams_on | 37.346 ms | 39.074 ms | 41.831 ms | 127.96 |
| sum | 20M | gpu_streams_off | 116.015 ms | 118.166 ms | 120.352 ms | 169.25 |
| sum | 20M | gpu_streams_on | 105.508 ms | 107.326 ms | 109.068 ms | 186.35 |
| count | 1M | gpu_streams_off | 13.078 ms | 13.436 ms | 13.813 ms | 74.43 |
| count | 1M | gpu_streams_on | 12.567 ms | 13.549 ms | 15.545 ms | 73.80 |
| count | 5M | gpu_streams_off | 37.442 ms | 37.757 ms | 38.183 ms | 132.42 |
| count | 5M | gpu_streams_on | 36.459 ms | 37.130 ms | 37.829 ms | 134.66 |
| count | 20M | gpu_streams_off | 115.246 ms | 116.139 ms | 117.240 ms | 172.21 |
| count | 20M | gpu_streams_on | 111.169 ms | 112.786 ms | 115.352 ms | 177.33 |
| avg | 1M | gpu_streams_off | 16.350 ms | 16.583 ms | 16.963 ms | 60.30 |
| avg | 1M | gpu_streams_on | 17.385 ms | 21.219 ms | 27.543 ms | 47.13 |
| avg | 5M | gpu_streams_off | 43.338 ms | 43.682 ms | 44.009 ms | 114.46 |
| avg | 5M | gpu_streams_on | 41.567 ms | 42.338 ms | 42.841 ms | 118.10 |
| avg | 20M | gpu_streams_off | 125.504 ms | 127.545 ms | 129.875 ms | 156.81 |
| avg | 20M | gpu_streams_on | 118.494 ms | 119.689 ms | 120.837 ms | 167.10 |
| min_max | 1M | gpu_streams_off | 15.782 ms | 15.957 ms | 16.197 ms | 62.67 |
| min_max | 1M | gpu_streams_on | 22.315 ms | 28.185 ms | 31.449 ms | 35.48 |
| min_max | 5M | gpu_streams_off | 43.644 ms | 44.156 ms | 44.879 ms | 113.23 |
| min_max | 5M | gpu_streams_on | 41.927 ms | 42.275 ms | 42.714 ms | 118.27 |
| min_max | 20M | gpu_streams_off | 123.205 ms | 124.593 ms | 126.425 ms | 160.52 |
| min_max | 20M | gpu_streams_on | 120.110 ms | 121.104 ms | 122.153 ms | 165.15 |
| combined | 1M | gpu_streams_off | 30.739 ms | 31.573 ms | 32.573 ms | 31.67 |
| combined | 1M | gpu_streams_on | 42.057 ms | 48.088 ms | 59.181 ms | 20.80 |
| combined | 5M | gpu_streams_off | 68.552 ms | 69.352 ms | 70.083 ms | 72.10 |
| combined | 5M | gpu_streams_on | 64.331 ms | 65.914 ms | 68.480 ms | 75.86 |
| combined | 20M | gpu_streams_off | 150.169 ms | 151.422 ms | 153.083 ms | 132.08 |
| combined | 20M | gpu_streams_on | 139.137 ms | 141.199 ms | 142.917 ms | 141.64 |
