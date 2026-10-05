"""Independent NumPy prototypes for unscaled Qwen2 and Gemma 1 models.

Native GGUF loading remains Llama-only; these explicit reference paths never
silently reinterpret other architectures as Llama. Configuration semantics follow
the installed Transformers Qwen2/Gemma implementations.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VariantConfig:
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    activation: str = "silu"

    def __post_init__(self):
        dims = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
        )
        if any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError("Variant dimensions must be positive integers")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Invalid rotary head dimension or GQA ratio")
        if self.architecture not in ("qwen2", "gemma"):
            raise ValueError("Unsupported variant")
        allowed = (
            ("silu",) if self.architecture == "qwen2" else ("gelu_pytorch_tanh", "gelu_new", "gelu")
        )
        if self.activation not in allowed or type(self.tie_word_embeddings) is not bool:
            raise ValueError("Invalid variant activation or tied-embedding flag")
        if (
            not np.isfinite(self.rms_norm_eps)
            or self.rms_norm_eps <= 0
            or not np.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise ValueError("Invalid variant numerical configuration")


def qwen_config(data):
    if data.get("model_type") != "qwen2":
        raise ValueError("Only Qwen2 is supported by this mapping")
    if (
        data.get("rope_scaling") is not None
        or data.get("use_sliding_window", False)
        or any(kind != "full_attention" for kind in data.get("layer_types") or [])
    ):
        raise ValueError("Scaled RoPE and sliding attention require separate implementations")
    if data.get("hidden_act", "silu") != "silu":
        raise ValueError("Qwen2 activation must be SiLU")
    hidden = data["hidden_size"]
    heads = data["num_attention_heads"]
    if type(hidden) is not int or type(heads) is not int or heads <= 0 or hidden % heads:
        raise ValueError("Qwen2 hidden width must divide into query heads")
    return VariantConfig(
        "qwen2",
        data["vocab_size"],
        hidden,
        data["intermediate_size"],
        data["num_hidden_layers"],
        heads,
        data.get("num_key_value_heads", heads),
        hidden // heads,
        data.get("max_position_embeddings", 32768),
        data.get("rms_norm_eps", 1e-6),
        data.get("rope_theta", 10000.0),
        data.get("tie_word_embeddings", False),
    )


def gemma_config(data):
    if data.get("model_type") != "gemma":
        raise ValueError("Only Gemma 1 is supported by this mapping")
    if data.get("rope_scaling") is not None or data.get("attention_bias", False):
        raise ValueError("Scaled RoPE and biased Gemma attention are unsupported")
    activation = data.get("hidden_act", "gelu_pytorch_tanh")
    if activation not in ("gelu_pytorch_tanh", "gelu_new", "gelu"):
        raise ValueError("Unsupported Gemma activation")
    return VariantConfig(
        "gemma",
        data["vocab_size"],
        data["hidden_size"],
        data["intermediate_size"],
        data["num_hidden_layers"],
        data["num_attention_heads"],
        data.get("num_key_value_heads", data["num_attention_heads"]),
        data.get("head_dim", 256),
        data.get("max_position_embeddings", 8192),
        data.get("rms_norm_eps", 1e-6),
        data.get("rope_theta", 10000.0),
        data.get("tie_word_embeddings", True),
        activation,
    )


def variant_norm(x, weight, config):
    from .math import rms_norm

    if config.architecture == "gemma":
        weight = np.asarray(weight, dtype=np.float32) + np.float32(1)
    return rms_norm(x, weight, config.rms_norm_eps)


def variant_embedding(table, tokens, config):
    result = np.asarray(table, dtype=np.float32)[tokens].copy()
    if config.architecture == "gemma":
        result *= np.float32(np.sqrt(config.hidden_size))
    return result


def variant_gate(gate, up, config):
    from .math import swiglu

    if config.activation == "silu":
        return swiglu(gate, up)
    gate = np.asarray(gate, dtype=np.float32)
    if config.activation == "gelu":
        import math

        erf = np.vectorize(math.erf, otypes=[np.float32])(gate / np.float32(np.sqrt(2)))
        activation = np.float32(0.5) * gate * (np.float32(1) + erf)
    else:
        inner = np.float32(np.sqrt(2 / np.pi)) * (gate + np.float32(0.044715) * gate**3)
        activation = np.float32(0.5) * gate * (np.float32(1) + np.tanh(inner))
    return (activation * np.asarray(up, dtype=np.float32)).astype(np.float32)


# VARIANT_CODE
