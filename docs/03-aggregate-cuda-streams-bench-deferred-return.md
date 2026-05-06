# Aggregate bench: streams_off vs streams_on after deferred-return pinned pool + pooled events

Run after fixing TLS destruction-order panic in `CuDFEvent::Drop` (`try_with` instead of `with`).

Setup: 4 worker partitions, `cuda_async_memory_resource` device pool, `pin_record_batch` enabled, deferred-return pinned pool, thread-local CUDA event pool.

## Median wall time (criterion estimate)

| Shape         | streams_off | streams_on | streams_on Δ |
|---------------|-------------|------------|--------------|
| sum/1M        | 11.87 ms    | 10.49 ms   | −11.6%       |
| sum/5M        | 33.75 ms    | 30.63 ms   | −9.3%        |
| sum/20M       | 113.26 ms   | 134.16 ms  | +18.5%       |
| count/1M      | 12.09 ms    | 11.88 ms   | −1.7%        |
| count/5M      | 33.89 ms    | 35.45 ms   | +4.6%        |
| count/20M     | 113.19 ms   | 120.25 ms  | +6.2%        |
| avg/1M        | 14.48 ms    | 13.25 ms   | −8.5%        |
| avg/5M        | 37.98 ms    | 36.05 ms   | −5.1%        |
| avg/20M       | 116.68 ms   | 116.09 ms  | −0.5%        |
| min_max/1M    | 14.68 ms    | 14.96 ms   | +1.9%        |
| min_max/5M    | 37.66 ms    | 40.81 ms   | +8.4%        |
| min_max/20M   | 116.39 ms   | 129.57 ms  | +11.3%       |
| combined/1M   | 23.63 ms    | 36.85 ms   | +56%         |
| combined/5M   | 50.43 ms    | 70.71 ms   | +40%         |
| combined/20M  | 136.24 ms   | 152.10 ms  | +11.6%       |

## Read

- Single-aggregation small shapes (1M, sum/avg) win by ~10% with streams.
- Larger single-aggregation shapes (20M) lose by ~10–20% with streams.
- The 5-aggregation `combined` shape is uniformly worse with streams; ~+50% at 1M/5M.
- The TLS fix held — bench completed cleanly.

## Why streams_on isn't a clean win at 4 partitions

With only 4 partitions the GPU has plenty of headroom to run partitions back-to-back on a single stream; cross-stream concurrency adds overhead (per-partition `CuDFStream` lookup in `CuDFTaskContext`, per-batch event record, deferred-return bookkeeping) without much overlap to amortize it.

The `combined` regression is the loudest: that shape runs aggregate→aggregate→...→aggregate, so the per-batch event/lookup overhead multiplies with depth.
