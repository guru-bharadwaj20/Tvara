# Quantized KV prototype

`QuantizedKV` stores signed int8 key and value lanes with one float32 scale per
token and head. For head dimension D, allocated bytes per token-head are
`2 * (D + 4)` versus `2 * D * 4` in F32. At D=32 the ratio is 0.28125.
There is no zero-point because the quantizer is symmetric around zero.

`tools/bench_quantized_kv.py` records actual encode duration, memory use,
maximum KV reconstruction error, causal attention error and a synthetic linear
logit projection's error/argmax agreement at 128, 512 and 4096 positions.
Run with `PYTHONPATH=python python tools/bench_quantized_kv.py`.
These synthetic checks do not establish pretrained-model quality or perplexity.
The prototype is not enabled in the native decoder; an integration must first
measure task quality, latency including dequantization, and multiple real models.
