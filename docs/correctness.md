# Correctness gates

Use deterministic float32 reference computation before implementing faster paths.
For each kernel, compare with independent NumPy expressions over ordinary, zero,
large, and boundary-sized inputs. Compare intermediate layer tensors to locate
errors before comparing final logits.

Initial float32 comparison targets are absolute tolerance 1e-5 and relative tolerance
1e-4 for tiny models. These are starting thresholds, not universal guarantees.
Any changed tolerance requires a recorded numerical justification. Quantized tests
compare the fast path to the same block format's reference dequantization; they do
not require quantized output to match full-precision weights exactly.

End-to-end checks include per-position logit error, top-1 agreement, deterministic
greedy token matches, and perplexity under matching tokenization and precision.
Store model revisions and fixture generation commands. Cache, batching, and
speculation paths must match their uncached or nonspeculative reference behavior.
Distribution testing supplements the speculative-sampling proof; it does not prove
exactness by itself. CI must run portable numerical regression tests on every change.
