# Architecture prototypes

The independent NumPy `VariantReferenceModel` supports unscaled full-attention
Qwen2 and Gemma 1 configurations. Qwen2 includes biased Q/K/V projections and
SiLU gating; Gemma uses offset RMSNorm weights, sqrt(hidden-size) embedding
scaling, independently specified head dimensions, and its configured GELU gate.
Both implement causal grouped-query attention and tied or untied output heads.

Tests compare normalization, rotary rotation and full tiny-model logits against
Transformers 4.57.1, with deterministic random weights and multiple prompts.
The Qwen tokenizer uses the SHA-pinned Qwen2.5 tokenizer artifact and independent
Rust token-ID comparisons for Unicode, chat markers and end-of-text markers.

These are reference prototypes, not native architecture support. GGUF loading
and native execution remain Llama-only. Qwen3, Qwen2-VL, Gemma 2/3, sliding
attention and scaled RoPE are explicitly rejected. No pretrained Qwen/Gemma
weight quality, generation throughput or accuracy claim follows from tiny-model
parity. Native integration requires architecture-aware tensor validation and
format mapping in addition to the reference mathematics.
