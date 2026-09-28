# Estimated implementation schedule

The original estimate is 18 weeks for four contributors, with overlapping work.
Week numbers are relative to implementation kickoff; they are not calendar dates
or evidence that work was completed during those weeks.

| Weeks | Kernels | Executor and caches | Decoding | Serving and evaluation |
|---|---|---|---|---|
| 1–2 | Toolchain and scalar design | NumPy model | Grammar/speculation study | Correctness harness |
| 3–5 | GGUF and quant references | Core model and tokenizer | Samplers and compiler | Baselines |
| 6–8 | AVX2 and workers | Paged KV | Automaton | Streaming API |
| 9–11 | Prefill and attention | Prefix tree | Cached masks | Load generator |
| 12–14 | Profiling | Continuous batching | Exact speculation | Templates and tools |
| 15–16 | Model variants | Preemption | Prompt lookup | Structured evaluation |
| 17–18 | Final measurements | Regression checks | Proof review | Documentation and release |

Only one implementation agent is currently assigned. Re-estimate after the first
validated milestone. Hardware-specific stretch tasks may extend beyond this plan.
Commit dates requested by the maintainer are metadata, not elapsed-work estimates.
