import numpy as np
import pytest
import torch
from transformers import GemmaConfig, GemmaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from tvara.reference.variants import gemma_config, qwen_config, VariantReferenceModel


@pytest.mark.parametrize(
    "architecture,tied", [("gemma", True), ("gemma", False), ("qwen2", True), ("qwen2", False)]
)
def test_variant_logits(architecture, tied):
    torch.manual_seed(246)
    common = dict(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    if architecture == "gemma":
        config = GemmaConfig(**common, head_dim=6, hidden_act="gelu_pytorch_tanh")
        reference = GemmaForCausalLM(config).eval()
        ours = gemma_config(config.to_dict())
    else:
        config = Qwen2Config(**common)
        reference = Qwen2ForCausalLM(config).eval()
        ours = qwen_config(config.to_dict())
    config._attn_implementation = "eager"
    model = VariantReferenceModel(
        ours, {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    )
    for tokens in ([1], [1, 2, 3, 4, 9], [3, 3, 3, 8, 1, 20, 7]):
        with torch.no_grad():
            expected = reference(torch.tensor([tokens]), use_cache=False).logits[0].numpy()
        np.testing.assert_allclose(model.forward(tokens), expected, rtol=2e-5, atol=2e-5)
    with pytest.raises(ValueError):
        model.forward([-1])
