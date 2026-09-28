# Prior work

These sources inform the design. Reimplement algorithms with attribution; do not
copy implementation code without respecting its license. No external source is
claimed as an implemented feature or as measured Tvara performance.

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): metadata and tensor storage format.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): matched CPU benchmark baseline.
- [llama2.c](https://github.com/karpathy/llama2.c): minimal transformer execution reference.
- [PagedAttention](https://arxiv.org/abs/2309.06180): paged KV ownership and storage.
- [SGLang](https://arxiv.org/abs/2312.07104): radix-tree prefix reuse.
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level batching.
- [Speculative decoding](https://arxiv.org/abs/2211.17192): exact rejection-based verification.
- [Speculative sampling](https://arxiv.org/abs/2302.01318): sampling distribution analysis.
- [FlashAttention](https://arxiv.org/abs/2205.14135): online softmax without storing full scores.
- [Outlines](https://arxiv.org/abs/2307.09702): constrained text generation.
- [XGrammar](https://arxiv.org/abs/2411.15100): vocabulary validity and mask caching.
- [JSONSchemaBench](https://arxiv.org/abs/2501.10868): structured-output evaluation.
- [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-1B): model architecture and weight conditions.
