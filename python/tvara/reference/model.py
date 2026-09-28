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


class ReferenceModel:
    def __init__(self, config, weights):
        self.config = config
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        d = config.hidden_size
        f = config.intermediate_size
        kv = d * config.num_key_value_heads // config.num_attention_heads
        required = {"model.embed_tokens.weight": (config.vocab_size, d), "model.norm.weight": (d,)}
        if not config.tie_word_embeddings:
            required["lm_head.weight"] = (config.vocab_size, d)
        for i in range(config.num_hidden_layers):
            for name, shape in {
                "input_layernorm.weight": (d,),
                "post_attention_layernorm.weight": (d,),
                "self_attn.q_proj.weight": (d, d),
                "self_attn.k_proj.weight": (kv, d),
                "self_attn.v_proj.weight": (kv, d),
                "self_attn.o_proj.weight": (d, d),
                "mlp.gate_proj.weight": (f, d),
                "mlp.up_proj.weight": (f, d),
                "mlp.down_proj.weight": (d, f),
            }.items():
                required[f"model.layers.{i}." + name] = shape
        for name, shape in required.items():
            if name not in self.weights or self.weights[name].shape != shape:
                raise ValueError(f"Missing or malformed model tensor: {name}")
            if not np.isfinite(self.weights[name]).all():
                raise ValueError(f"Nonfinite model tensor: {name}")

    @classmethod
    def from_directory(cls, path):
        from safetensors.numpy import load_file

        path = Path(path)
        return cls(
            Config.from_json(path / "config.json"), load_file(str(path / "model.safetensors"))
        )

    def forward(self, tokens):
        ids = np.asarray(tokens)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError("Tokens must be a nonempty integer sequence")
        if (
            ids.min() < 0
            or ids.max() >= self.config.vocab_size
            or len(ids) > self.config.max_position_embeddings
        ):
            raise ValueError("Token IDs or context length exceed model limits")
        x = self.weights["model.embed_tokens.weight"][ids].copy()
        positions = np.arange(len(ids))
        for i in range(self.config.num_hidden_layers):
            x = transformer_layer(x, self.weights, f"model.layers.{i}.", self.config, positions)
        x = rms_norm(x, self.weights["model.norm.weight"], self.config.rms_norm_eps)
        head = (
            self.weights["model.embed_tokens.weight"]
            if self.config.tie_word_embeddings
            else self.weights["lm_head.weight"]
        )
        return (x @ head.T).astype(np.float32)
