# Risks and stretch scope

| Risk | Mitigation and exit condition |
|---|---|
| Optimization hides arithmetic bugs | Keep independent scalar/reference comparisons |
| Tokenizer mismatch | Compare IDs and bytes on Unicode and special-token cases |
| Invalid binary metadata | Bound lengths/counts and reject overlapping or truncated storage |
| Cache use-after-free | Test refcounts, copy-on-write, eviction, and cancellation |
| Unfair CPU comparisons | Match files, thread counts, warm-ups, power state, and context |
| Grammar dead end | Report an exhausted allowed-token set instead of sampling invalid output |
| Speculative numerical instability | Test support mismatches, residual normalization, and rollback |
| Restricted model access | Keep offline tiny-model arithmetic tests; do not invent model results |

After the portable core passes, consider ARM NEON, AVX-512, KV quantization,
mixture-of-experts routing, tree speculation, and WebAssembly SIMD. Each stretch
feature needs a tested build or a documented feasibility assessment. Running an
x86 test does not establish ARM correctness; an unavailable machine leaves its
hardware validation task pending.
