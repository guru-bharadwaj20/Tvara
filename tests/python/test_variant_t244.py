import pytest
from tvara.reference.variants import gemma_config


def tiny():
    return dict(
        model_type="gemma",
        vocab_size=31,
        hidden_size=18,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        max_position_embeddings=64,
    )


def test_independent_head_dimension():
    c = gemma_config(tiny())
    assert c.head_dim * c.num_attention_heads != c.hidden_size
    assert c.tie_word_embeddings and c.activation == "gelu_pytorch_tanh"


@pytest.mark.parametrize(
    "change",
    [
        dict(model_type="gemma2"),
        dict(head_dim=3),
        dict(attention_bias=True),
        dict(hidden_act="silu"),
    ],
)
def test_rejects(change):
    with pytest.raises(ValueError):
        gemma_config({**tiny(), **change})
