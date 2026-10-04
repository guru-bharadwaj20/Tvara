# Radix prefix ownership

`RadixCache` stores compressed token edges and retains the physical KV blocks attached to each cached endpoint. Construct one instance per model revision, tokenizer revision, and tenant; the required isolation string documents that boundary. A caller must never attach KV from a different model or position policy. Mutations belong to one scheduler thread.

`lookup` reports structural overlap; `reuse` restores only an endpoint with an attached snapshot. Exact cached endpoints and partial overlap inside their edge are reusable. An internal branch without a snapshot currently returns a cold sequence, so some theoretically reusable prefixes remain misses. Full-block reuse is the default. Opt-in partial tails use `PagedCache` copy on write when appending, leaving retained prefix data unchanged.

Pin endpoints while scheduling requests; pins protect ancestors against leaf eviction. Touch cached leaves after access to maintain recency. Eviction releases only cache references; restored sequences remain independently owned. Reset rejects outstanding pins, clears retained snapshots, and preserves live sequences. Destroy the radix cache before its paged cache. Node IDs remain monotonic across reset.

The executable benchmark records actual measurement time, hit count and saved prefill tokens for a controlled synthetic workload. Saved tokens indicate avoided KV recomputation opportunities, not measured end-to-end inference speed. There is no cross-tenant or cross-model sharing and no claim of persistent cache compatibility.
