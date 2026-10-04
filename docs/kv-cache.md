# Paged KV cache

One block stores all layers' keys and values for a fixed number of token positions.
Its float32 payload size is layers * 2 * tokens_per_block * kv_heads * head_dim * 4
bytes. The pool preallocates a bounded number of blocks; table and allocator metadata
add memory overhead. Every reused block is zeroed before allocation.

Generation-tagged handles reject stale references. Sequence block tables map token
positions to physical blocks. Forking retains shared pages; appending into a shared
partial tail copies that page first. Full-prefix blocks stay shared. Truncation drops
whole rejected pages and hides rejected positions in the tail. Completion/cancellation
must drop the sequence to release its references. CacheExhausted leaves append state
unchanged and is intended for scheduler admission/preemption handling.

The cache is currently single-owner: a scheduler or caller must serialize mutation.
Audits verify free-list membership, block-table lengths, and reference counts; external
prefix pins must be supplied to the audit as extra expected references.

The paged native executor matches the uncached tiny-checkpoint oracle logits. It
currently materializes a layer's cached prefix for scalar attention, so paging correctness
is established without claiming fused attention performance. Later attention work will
consume pages directly. Run ctest --preset debug and the native-logit Python test to
reproduce the local cache and executor checks.
