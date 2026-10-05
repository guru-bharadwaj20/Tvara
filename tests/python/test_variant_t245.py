import pytest
import numpy as np

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm  # noqa: E402
from transformers.activations import ACT2FN  # noqa: E402
from tvara.reference.variants import gemma_config, variant_norm, variant_gate, variant_embedding  # noqa: E402


def config():
    return gemma_config(
        dict(
            model_type="gemma",
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
    )


def test_gemma_norm_and_activation():
    torch.manual_seed(245)
    norm = GemmaRMSNorm(16, eps=1e-6)
    x = torch.randn(7, 16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(16) - 0.5)
    np.testing.assert_allclose(
        variant_norm(x.numpy(), norm.weight.detach().numpy(), config()),
        norm(x).detach().numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    up = torch.randn(7, 16)
    np.testing.assert_allclose(
        variant_gate(x.numpy(), up.numpy(), config()),
        (ACT2FN["gelu_pytorch_tanh"](x) * up).numpy(),
        rtol=2e-6,
        atol=2e-6,
    )


def test_embedding_scale():
    table = np.arange(31 * 16, dtype=np.float32).reshape(31, 16)
    np.testing.assert_array_equal(variant_embedding(table, [1, 2], config()), table[[1, 2]] * 4)
