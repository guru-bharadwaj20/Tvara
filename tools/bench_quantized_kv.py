"""Measure int8 KV attention error and storage on deterministic synthetic inputs."""

import json
from pathlib import Path
import time
import sys
import numpy as np


def measure():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from tvara.quantized_kv import QuantizedKV
    from tvara.reference.math import attention

    rng = np.random.default_rng(252)
    rows = []
    for length in (128, 512, 4096):
        keys = rng.normal(size=(length, 2, 32)).astype(np.float32)
        values = rng.normal(size=keys.shape).astype(np.float32)
        query = rng.normal(size=(1, 4, 32)).astype(np.float32)
        projection = rng.normal(size=(128, 31)).astype(np.float32) * 0.1
        cache = QuantizedKV(2, 32, length)
        start = time.perf_counter_ns()
        for k, v in zip(keys, values, strict=True):
            cache.append(k, v)
        encode_us = (time.perf_counter_ns() - start) / 1000
        k, v = cache.decoded()
        original = attention(query, keys, values)
        quantized = attention(query, k, v)
        logits = original.reshape(1, -1) @ projection
        qlogits = quantized.reshape(1, -1) @ projection
        rows.append(
            dict(
                context=length,
                float_bytes=keys.nbytes + values.nbytes,
                int8_bytes=cache.active_bytes,
                memory_ratio=cache.active_bytes / (keys.nbytes + values.nbytes),
                maximum_kv_error=float(max(np.max(np.abs(keys - k)), np.max(np.abs(values - v)))),
                attention_max_error=float(np.max(np.abs(original - quantized))),
                projected_logit_max_error=float(np.max(np.abs(logits - qlogits))),
                projected_argmax_agreement=bool(logits.argmax() == qlogits.argmax()),
                encode_us=encode_us,
            )
        )
    return dict(
        scope="synthetic KV attention and linear logit projection; no pretrained-model perplexity claim",
        measurements=rows,
    )


def main():
    result = measure()
    path = Path("bench/results/variants/quantized-kv.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
