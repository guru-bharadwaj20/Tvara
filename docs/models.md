# Model targets

The intended production evaluation range is approximately 0.5–3 billion parameters.
Quantized weight size is only one part of memory usage: KV context, temporary
activations, tokenizer data, and mapped pages must also be measured.

| Priority | Model family | Sizes | Purpose |
|---|---|---|---|
| First | Llama 3.2 | 1B, later 3B | Establish executor correctness |
| Second | Qwen 2.5 | 0.5B, 1.5B, 3B | Small draft/target pairs and tokenizer coverage |
| Later | SmolLM2 | 1.7B | Additional architecture-compatible evaluation |
| Later | Gemma 3 | 1B | Architecture-specific normalization coverage |

Each actual run must pin model revision, tokenizer revision, GGUF checksum,
quantization, context length, and access/license conditions. These entries are
targets, not verified support. Tiny randomly initialized models are suitable for
offline arithmetic tests but do not establish pretrained-model quality or speed.
