import pytest
from tvara.reference.variants import qwen_config


def tiny():
    return dict(
        model_type="qwen2",
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


def test_mapping():
    c = qwen_config(tiny())
    assert (c.architecture, c.head_dim, c.num_key_value_heads) == ("qwen2", 4, 2)
    assert c.activation == "silu" and not c.tie_word_embeddings


@pytest.mark.parametrize(
    "change",
    [
        dict(model_type="qwen3"),
        dict(rope_scaling={"type": "linear", "factor": 2}),
        dict(use_sliding_window=True),
        dict(hidden_act="gelu"),
        dict(hidden_size=15),
        dict(num_attention_heads=0),
    ],
)
def test_rejects(change):
    with pytest.raises(ValueError):
        qwen_config({**tiny(), **change})
