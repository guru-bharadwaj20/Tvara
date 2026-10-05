import pytest
import numpy as np

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.models.qwen2.modeling_qwen2 import (  # noqa: E402
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import Qwen2Config  # noqa: E402
from tvara.reference.math import rms_norm, rope  # noqa: E402


def test_qwen_normalization():
    torch.manual_seed(242)
    norm = Qwen2RMSNorm(16, eps=1e-6)
    x = torch.randn(7, 16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(16))
    np.testing.assert_allclose(
        rms_norm(x.numpy(), norm.weight.detach().numpy()),
        norm(x).detach().numpy(),
        rtol=2e-6,
        atol=2e-6,
    )


def test_qwen_rotary():
    torch.manual_seed(242)
    config = Qwen2Config(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2, rope_theta=1000000.0
    )
    rotary = Qwen2RotaryEmbedding(config)
    q = torch.randn(1, 4, 7, 4)
    k = torch.randn(1, 2, 7, 4)
    positions = torch.arange(7)[None, :]
    cos, sin = rotary(q, positions)
    a, b = apply_rotary_pos_emb(q, k, cos, sin)
    np.testing.assert_allclose(
        rope(q[0].transpose(0, 1).numpy(), np.arange(7), config.rope_theta),
        a[0].transpose(0, 1).numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        rope(k[0].transpose(0, 1).numpy(), np.arange(7), config.rope_theta),
        b[0].transpose(0, 1).numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
