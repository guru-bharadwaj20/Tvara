# First reference model

Use hf-internal-testing/tiny-random-LlamaForCausalLM for the first independent
logit comparison. This is a random-weight Llama architecture checkpoint with a
16-wide hidden state and two layers. It is small enough to validate offline once
downloaded. It is not a useful pretrained text generator.

The production target remains Llama 3.2 1B after access conditions and architecture
variants are validated. Tiny-model comparisons establish arithmetic only, not
pretrained-model quality, tokenizer coverage, or production throughput.
