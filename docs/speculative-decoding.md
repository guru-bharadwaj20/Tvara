# Speculative sampling contract

Draft and target expose normalized conditional probabilities, owned token histories, and checkpoints. Tokenizer IDs, byte pieces, EOS policy, and tokenizer configuration fingerprint must match exactly. Sampling filters must be included in each model distribution before verification. Acceptance uses min(1, p(token)/q(token)); rejection samples normalized positive p-q mass. The all-accepted path samples an extra target token. Both model histories roll back to the accepted prefix and receive the same replacement or bonus token.

Proposal verification collects a batch of target conditionals in sequence. The reference and native batch paths execute target steps serially; it does not claim fused verification or native speedup. `SpeculativeModel` is an integration interface. Grammar constraints clone immutable UTF8/grammar state for draft and target branches, mask both distributions, and commit only emitted tokens. The grammar engine must contain a valid EOS ID equal to both model identities. A mask with zero allowed probability fails explicitly.

The independent distribution experiment checks acceptance algebra exactly and uses 200000 seeded samples per distribution pair, including disjoint supports, missing draft support, and equal models. An additional 10000 samples per pair exercise the actual speculative-step driver. Eighteen coordinate checks use a familywise Hoeffding bound at failure probability 0.001. This is an implementation sanity check, not a proof of all numerical behavior.

The overhead experiment reports high, moderate and zero-overlap synthetic probability workloads. It measures the Python orchestration cost against direct sampling, with actual execution timestamps. It does not measure trained-model text, coding or JSON task speed. The native sequence adapter supplies atomic C ABI batch decode and truncate operations; verification computes target steps sequentially under one mutex. Rollback recomputes the last retained token to recover its next-token logits. Optimized fused verification remains separate work; speculative decoding should be disabled when its measured overhead exceeds saved target work.


The native diagnostic runs the real C ABI decoder using the random-weight 257-token API fixture, with three text/code/JSON-shaped prompts. Both sides decode eight tokens with prefill setup excluded. The draft uses the same weights at temperature 0.5 and the target at 1.0, so this does not represent an independently trained small draft model. It records acceptance and measured decode speed without asserting useful generated content or production acceleration. The single-run Debug build result is a diagnostic, not a release performance claim.

## Final speedup measurement

`python tools/run_speculative_speedups.py` measures the native path with paired repetitions. The target is an eight-layer, 512-wide random-weight Llama fixture (257-token byte vocabulary). The draft is the same seeded fixture truncated to its first layer, so it shares embeddings, output projection and first block. Two regimes change only the output projection scale: "peaked" (std 0.5; first-token target entropy 0.0-1.3 bits) gives weakly agreeing draft and target distributions, and "flat" (std 0.05; about 7.1 bits) gives strongly agreeing ones. Arms are fixed draft lengths 1, 2 and 4, the adaptive `DraftController`, and prompt lookup with up to four tokens. Each arm runs four prompts (text, code, JSON and a repetitive string) for twelve seeds, emitting 32 tokens at temperature 1.0 with EOS disabled. The baseline samples the same target through the same Python/ctypes per-token path. Within each seed the two arms run back to back, alternating which runs first. The speedup is baseline time per token divided by speculative time per token. Each row reports the median over seeds with a 10,000-resample percentile-bootstrap 95% interval and an exact two-sided sign test.

Recorded result (`bench/results/speculative-speedups.json`, 2026-10-14 UTC, host B in [performance.md](performance.md); the two regimes ran in parallel single-threaded processes):

| Regime | Proposer | Acceptance | Tokens per target step | Speedup (range of prompt medians) |
|---|---|---|---|---|
| peaked | draft k=1 | 0.005 | 1.01 | 0.30-0.31 |
| peaked | draft k=2 | 0.003 | 1.01 | 0.23 |
| peaked | draft k=4 | 0.001 | 1.01 | 0.15-0.16 |
| peaked | draft adaptive | 0.005 | 1.01 | 0.28-0.29 |
| peaked | lookup k=4 | 0.264 | 1.35 | 0.42-0.60 |
| flat | draft k=1 | 0.477 | 1.47 | 0.51-0.53 |
| flat | draft k=2 | 0.350 | 1.69 | 0.35-0.43 |
| flat | draft k=4 | 0.214 | 1.82 | 0.26-0.30 |
| flat | draft adaptive | 0.345 | 1.53 | 0.42-0.44 |
| flat | lookup k=4 | 0.006 | 1.00 | 0.53-0.60 |

All 40 regime/arm/prompt rows were slower than plain decoding: every seed was slower (sign test p = 0.00049 for each row), and the largest upper confidence bound was 0.75. Plain decoding took about 11.5 ms per token in both regimes. The cause is structural, not tuning. `verify_batch` evaluates proposals one at a time, the replacement or bonus token is then decoded by `advance`, and a rejection re-decodes the last kept token on both models. A step that emits n tokens therefore costs at least n target decodes plus draft work, so speculation cannot save time until verification batches its target positions into one pass. Acceptance follows the expected pattern: it rises when distributions agree and falls as draft length grows. Prompt lookup is accepted when the low-entropy target repeats context.

Limits: random weights, a shared-weight draft, one host, and Python orchestration. The measurement does not predict acceptance for trained draft/target pairs or the speed of a batched verifier.
