# Iteration scheduler

The single-owner scheduler admits requests in stable priority order with aging every
eight iterations. Each admitted request reserves prompt plus maximum output tokens;
one bounded prefill chunk and one token per decode-ready request advance per step.
Finished and cancelled requests release native sequence ownership before admission.

Memory pressure drops the lowest-priority active KV state. Resumption replays prompt
and emitted tokens without emitting duplicates. The caller must resume preempted
requests after capacity becomes available; they are deliberately not silently retried.
This conservative token budget is an admission abstraction, not physical block accounting.

NativeSchedulerBackend executes real C++ sequences serially within decode batches.
It currently supports deterministic greedy generation and is separate from HTTP serving.
It does not claim parallel matrix batching. Scheduler tests use a deterministic backend
for lifecycle isolation; tools/benchmark_scheduler.py exercises the tiny native random
fixture and compares identical outputs for static versus continuous scheduling.
The recorded single-run timing is a smoke experiment, not a throughput recommendation.

## Concurrency latency results

`tools/run_scheduler_concurrency.py` submits N seeded requests together (16 prompt tokens, 16 output tokens, prefill chunk 8, `max_active = N`) to the native scheduler, with five repetitions per level after one discarded warm-up. Outputs must be identical across repetitions. Raw samples are in `bench/results/scheduler-concurrency.json` (measured 2026-10-14 UTC, `models/api-fixture/model.gguf`).

| Concurrency | TTFT p50 | TTFT p99 | Inter-token p50 | Inter-token p99 | Throughput (tok/s, median) |
|---|---|---|---|---|---|
| 1 | 1.32 ms | 1.62 ms | 0.104 ms | 0.256 ms | 5066 |
| 2 | 3.75 ms | 4.21 ms | 0.306 ms | 0.490 ms | 3702 |
| 4 | 7.48 ms | 11.57 ms | 0.582 ms | 0.873 ms | 3920 |
| 8 | 9.69 ms | 17.04 ms | 0.725 ms | 1.167 ms | 6204 |
| 16 | 19.89 ms | 29.67 ms | 1.433 ms | 4.235 ms | 5875 |

Latency grows roughly linearly with concurrency because decode is serial within each batch. Aggregate throughput stays roughly flat and noisy, so batching does not add throughput here. The fixture is a tiny random model, so absolute times show scheduler overhead, not realistic model cost. Percentiles pool samples across repetitions, and p99 rests on few samples at low concurrency.
