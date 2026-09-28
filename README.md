# Tvara

A CPU-only inference engine for small language models, built from scratch.

Tvara aims to load GGUF weights, execute quantized transformer models, and serve
text generation through an OpenAI-compatible API. The project focuses on
understanding each layer of inference and measuring correctness and performance
against Hugging Face Transformers and llama.cpp on the same hardware.

## Project status

Tvara is in the planning stage. This repository contains project documentation;
the engine, build system, server, and benchmarks have not been implemented yet.
There are no installation instructions or runnable commands at this stage.

## Planned capabilities

- GGUF loading with F16, Q4_0, Q4_K, and Q8_0 weight formats.
- Reference kernels followed by AVX2 optimization and CPU threading.
- Tokenization and a Llama-family forward pass, followed by Qwen and Gemma variants.
- Paged KV storage, radix-tree prefix reuse, and continuous batching.
- Draft-model and prompt-lookup speculative decoding.
- Grammar-constrained generation for JSON Schema and EBNF/GBNF.
- Streaming completions, chat templates, and tool calling.

These are planned features, not claims of current support. The core language
(C++20 or Rust) and serving stack will be decided during project setup. AVX-512,
ARM NEON, and other portability work are later goals.

## Development approach

Start with a scalar or NumPy reference and establish correctness before optimizing.
Compare logits, greedy tokens, and perplexity with reference implementations.
Measure prefill and decode throughput, memory bandwidth, time to first token,
inter-token latency, and structured-output overhead under reproducible conditions.
No performance results are available yet.

The proposal estimates 18 weeks for a team of four. This is a planning target,
not a delivery commitment; overlapping work depends on staffing and completed
prerequisites.

## Documentation and contributions

- [Project proposal](Tvara%20Project%20Idea.md): architecture, evaluation, references,
  and stretch goals.
- [Contributing guide and timeline](CONTRIBUTING.md): workflow and task status.

See the contributing guide before starting work. Model weights, datasets, and
third-party dependencies retain their own licenses and access requirements.

## License

Tvara's original code and documentation are available under the
[MIT License](LICENSE).
