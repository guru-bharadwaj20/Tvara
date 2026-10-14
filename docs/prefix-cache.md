# Radix prefix ownership

`RadixCache` stores compressed token edges and retains the physical KV blocks attached to each cached endpoint. Construct one instance per model revision, tokenizer revision, and tenant; the required isolation string documents that boundary. A caller must never attach KV from a different model or position policy. Mutations belong to one scheduler thread.

`lookup` reports structural overlap; `reuse` restores only an endpoint with an attached snapshot. Exact cached endpoints and partial overlap inside their edge are reusable. An internal branch without a snapshot currently returns a cold sequence, so some theoretically reusable prefixes remain misses. Full-block reuse is the default. Opt-in partial tails use `PagedCache` copy on write when appending, leaving retained prefix data unchanged.

Pin endpoints while scheduling requests; pins protect ancestors against leaf eviction. Touch cached leaves after access to maintain recency. Eviction releases only cache references; restored sequences remain independently owned. Reset rejects outstanding pins, clears retained snapshots, and preserves live sequences. Destroy the radix cache before its paged cache. Node IDs remain monotonic across reset.

The executable benchmark records actual measurement time, hit count and saved prefill tokens for a controlled synthetic workload. Saved tokens indicate avoided KV recomputation opportunities, not measured end-to-end inference speed. There is no cross-tenant or cross-model sharing and no claim of persistent cache compatibility.

## Conversation trace results

`tools/run_prefix_cache_traces.py` replays four seeded conversations of five turns (64-token shared system prompt, 16-48 token user turns, 12-token greedy replies) through the native engine, alternating cache-off and cache-on runs for three repetitions. With the cache on, each conversation keeps its live native sequence between turns, so only the new suffix is prefilled. Results are in `bench/results/prefix-cache-traces.json` (measured 2026-10-14 UTC, Intel Family 6 Model 61, `models/tiny-llama/reference.gguf`).

| Metric | Value |
|---|---|
| Prompt tokens over 20 turns | 3675 |
| Reused prompt tokens | 2765 (75.2%) |
| Turns with reuse | 16 of 20 (each conversation's first turn is cold) |
| Median total prefill, cache off | 3.47 s |
| Median total prefill, cache on | 0.98 s |
| Median prefill speedup | 3.55x |
| Greedy replies and logits identical | yes |

Limitations: the model is a tiny random-weight fixture and the token IDs are synthetic, so these are prefill-cost ratios, not quality or chat-latency results. Reuse is per conversation only, because the native C API has no sequence fork (it gained `tvara_sequence_truncate` but cannot share one prefix between sequences), so the shared system prompt is not reused across conversations. Single machine, single thread, no thermal or background-load control.
