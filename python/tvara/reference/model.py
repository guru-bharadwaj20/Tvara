"""Independent Llama forward computation using NumPy only."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .math import attention, rms_norm, rope, swiglu


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    def __post_init__(self):
        dims = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError("Model dimensions must be positive integers")
        if (
            self.hidden_size % self.num_attention_heads
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Invalid head dimensions or GQA ratio")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if (
            not np.isfinite(self.rms_norm_eps)
            or self.rms_norm_eps <= 0
            or not np.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise ValueError("Invalid normalization or RoPE configuration")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("model_type", "llama") != "llama" or data.get("rope_scaling") is not None:
            raise ValueError("Only unscaled Llama RoPE is supported by this reference")
        if data.get("attention_bias", False) or data.get("hidden_act", "silu") != "silu":
            raise ValueError("Biased attention and non-SiLU activations are unsupported")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def transformer_layer(x, weights, prefix, config, positions):
    def w(name):
        return weights[prefix + name]

    h, d = config.num_attention_heads, config.hidden_size // config.num_attention_heads
    norm = rms_norm(x, w("input_layernorm.weight"), config.rms_norm_eps)
    q = (norm @ w("self_attn.q_proj.weight").T).reshape(-1, h, d)
    k = (norm @ w("self_attn.k_proj.weight").T).reshape(-1, config.num_key_value_heads, d)
    v = (norm @ w("self_attn.v_proj.weight").T).reshape(-1, config.num_key_value_heads, d)
    q, k = rope(q, positions, config.rope_theta), rope(k, positions, config.rope_theta)
    mixed = attention(q, k, v).reshape(-1, config.hidden_size)
    x = x + mixed @ w("self_attn.o_proj.weight").T
    norm = rms_norm(x, w("post_attention_layernorm.weight"), config.rms_norm_eps)
    gated = swiglu(norm @ w("mlp.gate_proj.weight").T, norm @ w("mlp.up_proj.weight").T)
    return (x + gated @ w("mlp.down_proj.weight").T).astype(np.float32)
