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
