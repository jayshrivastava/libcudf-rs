# Aggregate CUDA Streams Benchmark Results

Run timestamp: 2026-05-03T03:21:13Z

GPU: Tesla T4, 15360 MiB

Command:

```text
cargo bench -p libcudf-datafusion --bench aggregate -- gpu_streams --sample-size 10 --warm-up-time 1 --measurement-time 5
```

This was a short Criterion run filtered to the aggregate benchmark's CUDA variants:

- `gpu_streams_off`
- `gpu_streams_on`

The table uses Criterion's middle time estimate. Speedup is `streams_off_time / streams_on_time`, so values above `1.00x` mean streams were faster.

| Benchmark | Rows | Streams off | Streams on | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `sum` | 1,000,000 | 15.274 ms | 14.760 ms | 1.03x |
| `sum` | 5,000,000 | 40.562 ms | 37.672 ms | 1.08x |
| `sum` | 20,000,000 | 122.80 ms | 114.12 ms | 1.08x |
| `count` | 1,000,000 | 16.864 ms | 19.409 ms | 0.87x |
| `count` | 5,000,000 | 39.538 ms | 36.896 ms | 1.07x |
| `count` | 20,000,000 | 126.33 ms | 117.31 ms | 1.08x |
| `avg` | 1,000,000 | 22.041 ms | 22.034 ms | 1.00x |
| `avg` | 5,000,000 | 44.607 ms | 41.568 ms | 1.07x |
| `avg` | 20,000,000 | 128.01 ms | 124.97 ms | 1.02x |
| `min_max` | 1,000,000 | 20.174 ms | 26.134 ms | 0.77x |
| `min_max` | 5,000,000 | 46.096 ms | 40.929 ms | 1.13x |
| `min_max` | 20,000,000 | 131.39 ms | 124.05 ms | 1.06x |
| `combined` | 1,000,000 | 40.452 ms | 51.591 ms | 0.78x |
| `combined` | 5,000,000 | 65.905 ms | 61.910 ms | 1.06x |
| `combined` | 20,000,000 | 157.69 ms | 153.46 ms | 1.03x |

Summary:

- Streams helped most at 5M rows: roughly `1.06x` to `1.13x` across all aggregate shapes.
- Streams were consistently positive at 20M rows, but the gains were smaller for `avg` and `combined`.
- Streams hurt small 1M-row `count`, `min_max`, and `combined` runs, likely because stream setup and task-context overhead dominate the limited GPU work.
- Best observed result: `min_max` at 5M rows, `1.13x` faster with streams.
