# Tvara

A CPU-only inference engine for small language models, built from scratch.

Tvara aims to load GGUF weights, execute quantized transformer models, and serve
text generation through an OpenAI-compatible API. The project focuses on
understanding each layer of inference and measuring correctness and performance
against Hugging Face Transformers and llama.cpp on the same hardware.

## Purpose and scope

Build and understand a complete local inference pipeline on CPUs. Start with a
single Llama-family sequence and a float32 reference; add quantization, concurrency,
and constrained decoding only after correctness is established. The first target
is x86-64 with a scalar fallback and optional AVX2. GPU execution and model training
are outside the scope. This project reports measured results rather than assumed
speedups or promised parity with mature engines.

## Project status

Tvara is in early implementation. The portable C++20 build and native GGUF loader
are available, and the independent NumPy Llama reference passes pinned tiny-model
logit and greedy-token checks. The scalar native model executor now passes tiny-checkpoint logit checks. A local completions API and SSE streaming are available; native chat templates,
optimized execution, and final benchmarks remain under development.

See [local development](docs/development.md) for exercised build/test commands and
[reference validation](docs/reference-validation.md) for numerical coverage.

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

## Local development

See [development commands](docs/development.md), [native bindings](docs/native-bindings.md),
and [API setup](docs/api.md). Current inference validation uses tiny/random fixtures;
no pretrained throughput or quality results are claimed.

## Documentation and contributions

- [Architecture proposal](docs/design.md), [model targets](docs/models.md),
  [correctness gates](docs/correctness.md), and [benchmark protocol](docs/benchmarks.md).
- [Local development](docs/development.md) and the
  [pull request template](.github/pull_request_template.md) describe how changes are
  validated.

Model weights, datasets, and third-party dependencies retain their own licenses and
access requirements.

## License

Tvara's original code and documentation are available under the
[MIT License](LICENSE).
