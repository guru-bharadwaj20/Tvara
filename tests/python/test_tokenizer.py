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


def test_base_interface_rejects_invalid_ids():
    from tvara.tokenizer import Tokenizer

    class Literal(Tokenizer):
        def encode(self, text, allow_special=False):
            return [self.ids[text]]

        def token_bytes(self, token):
            return self.vocabulary[token].encode()

    t = Literal(["a", "b"])
    assert t.decode([0, 1]) == "ab"
    with pytest.raises(ValueError):
        t.decode([-1])


def test_byte_mapping_is_bijective_over_all_bytes():
    from tvara.tokenizer import byte_mapping

    forward, reverse = byte_mapping()
    assert len(forward) == len(reverse) == 256
    assert bytes(reverse[forward[b]] for b in range(256)) == bytes(range(256))
    assert forward[32] == chr(0x120)
