"""Experimental per-token, per-head symmetric int8 KV storage.

Not integrated into the native decoder. Every token-head receives its own float32
scale. Error measurements describe synthetic attention, not model perplexity.
"""

import numpy as np


class QuantizedKV:
    def __init__(self, heads, dimension, capacity):
        if any(type(v) is not int or v <= 0 for v in (heads, dimension, capacity)):
            raise ValueError("KV dimensions and capacity must be positive integers")
        self.keys = np.empty((capacity, heads, dimension), dtype=np.int8)
        self.values = np.empty_like(self.keys)
        self.key_scales = np.empty((capacity, heads, 1), dtype=np.float32)
        self.value_scales = np.empty_like(self.key_scales)
        self.length = 0

    @staticmethod
    def encode(row):
        row = np.asarray(row, dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError("Nonfinite KV input")
        maximum = np.max(np.abs(row), axis=-1, keepdims=True)
        scales = np.where(maximum > 0, maximum / np.float32(127), np.float32(1)).astype(np.float32)
        if (scales == 0).any():
            scales = np.maximum(scales, np.nextafter(np.float32(0), np.float32(1)))
        values = np.clip(np.rint(row / scales), -127, 127).astype(np.int8)
        return values, scales

    def append(self, key, value):
        key = np.asarray(key, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        if key.shape != self.keys.shape[1:] or value.shape != key.shape:
            raise ValueError("Malformed KV token shape")
        if self.length >= len(self.keys):
            raise ValueError("Quantized KV capacity exceeded")
        k, ks = self.encode(key)
        v, vs = self.encode(value)
        self.keys[self.length] = k
        self.key_scales[self.length] = ks
        self.values[self.length] = v
        self.value_scales[self.length] = vs
        self.length += 1

    def decoded(self):
        n = self.length
        return (
            self.keys[:n].astype(np.float32) * self.key_scales[:n],
            self.values[:n].astype(np.float32) * self.value_scales[:n],
        )

    @property
    def allocated_bytes(self):
        return sum(a.nbytes for a in (self.keys, self.values, self.key_scales, self.value_scales))

    @property
    def active_bytes(self):
        return self.allocated_bytes * self.length // len(self.keys)
