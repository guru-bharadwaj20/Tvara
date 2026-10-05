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


class VariantReferenceModel:
    def __init__(self, config, weights):
        self.config = config
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        h, d = config.hidden_size, config.head_dim
        q = config.num_attention_heads * d
        kv = config.num_key_value_heads * d
        f = config.intermediate_size
        required = {"model.embed_tokens.weight": (config.vocab_size, h), "model.norm.weight": (h,)}
        if not config.tie_word_embeddings:
            required["lm_head.weight"] = (config.vocab_size, h)
        for i in range(config.num_hidden_layers):
            prefix = f"model.layers.{i}."
            shapes = {
                "input_layernorm.weight": (h,),
                "post_attention_layernorm.weight": (h,),
                "self_attn.q_proj.weight": (q, h),
                "self_attn.k_proj.weight": (kv, h),
                "self_attn.v_proj.weight": (kv, h),
                "self_attn.o_proj.weight": (h, q),
                "mlp.gate_proj.weight": (f, h),
                "mlp.up_proj.weight": (f, h),
                "mlp.down_proj.weight": (h, f),
            }
            if config.architecture == "qwen2":
                shapes.update(
                    {
                        "self_attn.q_proj.bias": (q,),
                        "self_attn.k_proj.bias": (kv,),
                        "self_attn.v_proj.bias": (kv,),
                    }
                )
            required.update({prefix + k: v for k, v in shapes.items()})
        for name, shape in required.items():
            if (
                name not in self.weights
                or self.weights[name].shape != shape
                or not np.isfinite(self.weights[name]).all()
            ):
                raise ValueError(f"Missing, malformed or nonfinite variant weight: {name}")

    def forward(self, tokens):
        from .math import attention, rope

        c = self.config
        ids = np.asarray(tokens)
        if (
            ids.ndim != 1
            or not ids.size
            or ids.dtype.kind not in "iu"
            or ids.min() < 0
            or ids.max() >= c.vocab_size
            or len(ids) > c.max_position_embeddings
        ):
            raise ValueError("Invalid variant token sequence")
        x = variant_embedding(self.weights["model.embed_tokens.weight"], ids, c)
        positions = np.arange(len(ids))
        for i in range(c.num_hidden_layers):
            prefix = f"model.layers.{i}."

            def w(name):
                return self.weights[prefix + name]

            norm = variant_norm(x, w("input_layernorm.weight"), c)

            def project(name, heads):
                value = norm @ w(f"self_attn.{name}_proj.weight").T
                if c.architecture == "qwen2":
                    value += w(f"self_attn.{name}_proj.bias")
                return value.reshape(len(ids), heads, c.head_dim)

            q = rope(project("q", c.num_attention_heads), positions, c.rope_theta)
            k = rope(project("k", c.num_key_value_heads), positions, c.rope_theta)
            v = project("v", c.num_key_value_heads)
            mixed = attention(q, k, v).reshape(len(ids), c.num_attention_heads * c.head_dim)
            x = x + mixed @ w("self_attn.o_proj.weight").T
            norm = variant_norm(x, w("post_attention_layernorm.weight"), c)
            gated = variant_gate(
                norm @ w("mlp.gate_proj.weight").T, norm @ w("mlp.up_proj.weight").T, c
            )
            x = (x + gated @ w("mlp.down_proj.weight").T).astype(np.float32)
        x = variant_norm(x, self.weights["model.norm.weight"], c)
        head = (
            self.weights["model.embed_tokens.weight"]
            if c.tie_word_embeddings
            else self.weights["lm_head.weight"]
        )
        return (x @ head.T).astype(np.float32)


# VARIANT_CODE
