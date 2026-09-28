# Reference validation

The local fixture uses the checkpoint revision and SHA-256 values in
tests/fixtures/reference-model.json. The Hugging Face oracle runs on CPU in float32,
uses eager attention, disables KV caching, fixes the Torch seed to zero, and limits
Torch to one worker. NumPy independently computes RMSNorm, split-half RoPE, causal
GQA, SwiGLU, residuals, final normalization, and projection.

Full per-position logits for three fixed token prompts are stored in compressed
NumPy fixtures. Comparison requires atol=1e-5, rtol=1e-4, and exact top-1 agreement.
Six generated greedy tokens are checked separately. Perplexity tests use known
uniform and shifted distributions; no WikiText-2 result is claimed yet.

Reproduce with the oracle optional dependencies and the pinned local checkpoint:

```text
python tools/capture_reference.py
python tools/capture_greedy.py
python -m pytest -q tests/python
```

Current coverage is unscaled Llama RoPE without attention biases. No pretrained
Llama 3.2 weights, quantized-model perplexity, or llama.cpp parity has been tested.
Optional checkpoint tests skip when weights are absent; that skip is not a passing
real-model evaluation. Model files remain ignored and are not redistributed.
