import pytest
from tvara.tokenizer import TokenizerMetadata


def test_metadata_rejects_duplicates_and_invalid_special_ids():
    metadata = {"tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.bos_token_id": 0}
    assert TokenizerMetadata.from_mapping(metadata).bos_id == 0
    metadata["tokenizer.ggml.bos_token_id"] = 2
    with pytest.raises(ValueError):
        TokenizerMetadata.from_mapping(metadata)
    with pytest.raises(ValueError):
        TokenizerMetadata.from_mapping({"tokenizer.ggml.tokens": ["a", "a"]})
