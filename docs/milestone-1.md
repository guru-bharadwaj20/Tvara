# Milestone 1: portable correctness foundation

Implementation owner: the coding agent working under guru-bharadwaj20's direction.
Maintainer and acceptance reviewer: guru-bharadwaj20. No additional contributors
are assigned. Follow the maintainer's authorized task scope; do not interpret this
assignment as permission to fabricate acceptance or benchmark results.

Acceptance criteria:

1. A clean C++20 debug and release build runs native tests on x86-64.
2. Python reference kernels pass independent numerical tests.
3. A tiny offline Llama-family model matches Hugging Face float32 logits.
4. GGUF parsing rejects malformed and truncated fixture files.
5. Scalar quantized kernels match reference dequantization.
6. Local setup and CI commands are documented and exercised.

This milestone excludes real-model throughput claims, serving readiness, and SIMD
speed claims. A restricted pretrained model download or unavailable target machine
is recorded explicitly and leaves the corresponding evaluation task pending.
