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


def test_bpe_rank_order_and_overlapping_pairs():
    from tvara.tokenizer import merge_bpe

    assert merge_bpe("abc", {("b", "c"): 0, ("a", "b"): 1}) == ["a", "bc"]
    assert merge_bpe("aaaa", {("a", "a"): 0, ("aa", "aa"): 1}) == ["aaaa"]
    assert merge_bpe("xyz", {}) == ["x", "y", "z"]


def byte_fixture():
    from tvara.tokenizer import ByteBPE, byte_mapping

    forward, _ = byte_mapping()
    words = list(forward.values())
    words += ["ab", "abc"]
    return ByteBPE(words, [("a", "b"), ("ab", "c")])


def test_byte_bpe_roundtrip_and_merges():
    t = byte_fixture()
    assert t.encode("abc") == [t.ids["abc"]]
    for text in [" a b", "hello\nworld", "é", "भारत", "😀"]:
        assert t.decode(t.encode(text)) == text


def sentencepiece_fixture():
    from tvara.tokenizer import SentencePiece

    m = TokenizerMetadata(
        "unigram",
        ("<unk>", chr(0x2581), "a", "b", chr(0x2581) + "ab"),
        (-10.0, -1.0, -1.0, -1.0, -0.1),
        (2, 1, 1, 1, 1),
    )
    return SentencePiece(m)


def test_sentencepiece_vocab_retains_scores_types_and_unknown_id():
    t = sentencepiece_fixture()
    assert t.unknown == 0 and t.metadata.scores[-1] == -0.1
    assert t.ids[chr(0x2581) + "ab"] == 4


def test_unigram_chooses_the_highest_scoring_segmentation():
    t = sentencepiece_fixture()
    assert t.encode("ab") == [4]
    assert t.encode("a") == [1, 2]
    assert t.encode("") == []
    assert t.encode("z") == [1, 0]


def test_sentencepiece_byte_fallback_decodes_multibyte_character():
    from tvara.tokenizer import SentencePiece

    words = ("<unk>", chr(0x2581)) + tuple(f"<0x{i:02X}>" for i in range(256))
    m = TokenizerMetadata("unigram", words, tuple([-1.0] * len(words)), (2, 1) + tuple([6] * 256))
    t = SentencePiece(m)
    assert t.decode(t.encode("😀")) == "😀"


def test_special_tokens_are_opt_in_and_prefer_longer_matches():
    from tvara.tokenizer import ByteBPE

    t = byte_fixture()
    words = list(t.vocabulary) + ["<end>", "<end>long"]
    t = ByteBPE(words, [("a", "b"), ("ab", "c")], special_ids=[len(words) - 2, len(words) - 1])
    assert t.encode("abc<end>long", allow_special=True) == [t.ids["abc"], t.ids["<end>long"]]
    assert t.ids["<end>"] not in t.encode("<end>")
    assert t.decode([t.ids["<end>"]], skip_special=True) == ""
