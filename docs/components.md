# Component contracts

| Component | Input | Output and ownership |
|---|---|---|
| GGUF reader | File path | Validated metadata and immutable tensor storage |
| Tokenizer | Unicode text or token IDs | Token IDs or decoded byte stream |
| Kernels | Validated dimensions and buffers | Caller-owned computed outputs |
| Executor | Model config, tokens, positions, KV view | Logits and newly appended KV |
| KV allocator | Block requests and releases | Reference-counted physical blocks |
| Prefix cache | Prompt token sequences | Pinned reusable prefix references |
| Scheduler | Requests and cache capacity | Bounded prefill/decode iterations |
| Grammar | Schema/grammar and emitted bytes | Allowed-token masks and acceptance state |
| Sampler | Logits, mask, parameters, RNG | Next token or explicit exhausted-mask error |
| Speculation | Draft and target distributions | Exact verified tokens and rollback point |
| API adapter | Validated client requests | Completion objects or SSE events |

Invalid files and incompatible configurations fail before execution. Array shapes
are checked at public boundaries. Numerical kernels do not parse model metadata.
The API must propagate cancellation and avoid exposing filesystem details in errors.
