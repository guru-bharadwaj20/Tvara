import numpy as np
import pytest
from tvara.reference.model import Config, transformer_layer


def tiny_config():
    return Config(16, 8, 12, 1, 2, 1, 32)


def zero_layer(config):
    d = config.hidden_size
    kv = d * config.num_key_value_heads // config.num_attention_heads
    sizes = {
        "self_attn.q_proj.weight": (d, d),
        "self_attn.k_proj.weight": (kv, d),
        "self_attn.v_proj.weight": (kv, d),
        "self_attn.o_proj.weight": (d, d),
        "mlp.gate_proj.weight": (config.intermediate_size, d),
        "mlp.up_proj.weight": (config.intermediate_size, d),
        "mlp.down_proj.weight": (d, config.intermediate_size),
    }
    w = {n: np.zeros(shape, dtype=np.float32) for n, shape in sizes.items()}
    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        w[n] = np.ones(d, dtype=np.float32)
    return w


def test_zero_projections_leave_layer_residual_unchanged():
    c = tiny_config()
    x = np.arange(24, dtype=np.float32).reshape(3, 8)
    np.testing.assert_array_equal(transformer_layer(x, zero_layer(c), "", c, [0, 1, 2]), x)
    with pytest.raises(ValueError):
        Config(16, 7, 12, 1, 2, 1)


def test_model_validates_weights_tokens_and_context():
    from tvara.reference.model import ReferenceModel

    c = tiny_config()
    w = {"model.layers.0." + k: v for k, v in zero_layer(c).items()}
    w["model.embed_tokens.weight"] = np.ones((16, 8), dtype=np.float32)
    w["model.norm.weight"] = np.ones(8, dtype=np.float32)
    w["lm_head.weight"] = np.ones((16, 8), dtype=np.float32)
    model = ReferenceModel(c, w)
    assert model.forward([1, 2, 3]).shape == (3, 16)
    for ids in ([], [16], [-1], [1.0], list(range(16)) * 3):
        with pytest.raises(ValueError):
            model.forward(ids)
    with pytest.raises(ValueError):
        ReferenceModel(c, {})
