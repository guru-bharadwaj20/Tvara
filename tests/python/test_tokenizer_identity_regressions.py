from pathlib import Path
import threading
import numpy as np
import pytest
from tvara.native import NativeModel
from tvara.speculative import (
    NativeSpeculativeModel,
    TokenizerIdentity,
    tokenizer_identity,
    tokenizer_fingerprint,
)
from tvara.tokenizer import ByteBPE, SentencePiece, TokenizerMetadata, byte_mapping

ROOT = Path(__file__).resolve().parents[2]


class MetadataModel(NativeModel):
    def __init__(self, metadata):
        self._lock = threading.RLock()
        self._tokenizer = None
        self._metadata = metadata
        self.vocab_size = len(metadata["tokenizer.ggml.tokens"])
        self._handle = 1

    @property
    def metadata(self):
        return self._metadata


def fixture():
    words = tuple(byte_mapping()[0][i] for i in range(256)) + ("<eos>",)
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": list(words),
        "tokenizer.ggml.eos_token_id": 256,
    }
    return MetadataModel(metadata), ByteBPE(words, [], special_ids=(256,))


def test_attachment_checks_exact_raw_vocabulary_order_without_mutating_previous_attachment():
    model, good = fixture()
    model.attach_tokenizer(good)
    words = list(good.vocabulary)
    words[0], words[1] = words[1], words[0]
    wrong = ByteBPE(words, [], special_ids=(256,))
    with pytest.raises(ValueError, match="ordered"):
        model.attach_tokenizer(wrong)
    assert model.encode("\x00") == [0]
    assert model._tokenizer is good


def test_model_control_token_status_and_merge_ranks_are_checked():
    model, good = fixture()
    with pytest.raises(ValueError, match="special"):
        model.attach_tokenizer(ByteBPE(good.vocabulary, []))
    words = tuple(byte_mapping()[0][i] for i in range(256)) + ("ab", "bc")
    model = MetadataModel({"tokenizer.ggml.tokens": list(words), "tokenizer.ggml.merges": ["a b"]})
    model.attach_tokenizer(ByteBPE(words, ["a b"]))
    with pytest.raises(ValueError, match="merge"):
        model.attach_tokenizer(ByteBPE(words, ["b c"]))


def test_model_inventory_decodes_byte_unicode_and_excludes_eos():
    model, tokenizer = fixture()
    assert model.token_inventory() == tuple(bytes([i]) for i in range(256)) + (b"",)
    model.attach_tokenizer(tokenizer)
    assert model.token_inventory() == tuple(bytes([i]) for i in range(256)) + (b"",)


def test_complete_identity_distinguishes_all_bpe_encoding_settings():
    words = tuple(byte_mapping()[0][i] for i in range(256)) + ("ab", "<eos>")
    base = ByteBPE(words, [], special_ids=(257,))
    identities = [tokenizer_identity(base, 257)]
    for changed in (
        ByteBPE(words, ["a b"], special_ids=(257,)),
        ByteBPE(words, [], normalizer="NFC", special_ids=(257,)),
        ByteBPE(words, [], pattern=r"\S+", special_ids=(257,)),
        ByteBPE(words, [], special_ids=(257,), add_prefix_space=True),
        ByteBPE(words, [], special_ids=(256, 257)),
    ):
        identities.append(tokenizer_identity(changed, 257))
    assert len({tokenizer_fingerprint(i) for i in identities}) == len(identities)
    assert identities[0].pieces[-1] == b""
    with pytest.raises(ValueError, match="EOS"):
        tokenizer_identity(base, 0)


def test_native_adapter_rejects_wrong_inventory_before_allocating_sequence():
    model, _ = fixture()
    wrong = tuple(bytes([255 - i]) for i in range(256)) + (b"",)
    with pytest.raises(ValueError, match="token bytes"):
        NativeSpeculativeModel(model, [1], TokenizerIdentity(wrong, "bad-fixture", 256))


def test_native_adapter_rejects_forged_configuration_for_attached_tokenizer():
    model, tokenizer = fixture()
    model.attach_tokenizer(tokenizer)
    with pytest.raises(ValueError, match="tokenizer_identity"):
        NativeSpeculativeModel(
            model, [1], TokenizerIdentity(model.token_inventory(), "same-arbitrary-label", 256)
        )


def test_sentencepiece_identity_captures_scores_and_whitespace_settings():
    metadata = TokenizerMetadata.from_mapping(
        {
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["a", "<eos>"],
            "tokenizer.ggml.token_type": [1, 3],
            "tokenizer.ggml.scores": [0.2, 0],
            "tokenizer.ggml.eos_token_id": 1,
        }
    )
    one = tokenizer_identity(SentencePiece(metadata), 1)
    two = tokenizer_identity(SentencePiece(metadata, remove_extra_whitespace=False), 1)
    from dataclasses import replace

    three = tokenizer_identity(SentencePiece(replace(metadata, scores=(0.3, 0))), 1)
    assert one.pieces == two.pieces == three.pieces
    assert len({tokenizer_fingerprint(i) for i in (one, two, three)}) == 3


def test_real_native_adapter_accepts_derived_api_identity_and_rejects_reordered_pieces():
    import os

    real_root = Path(os.environ.get("TVARA_AUDIT_ROOT", ROOT))
    from tvara.native import find_library

    try:
        binary = find_library()
    except FileNotFoundError:
        pytest.skip("Native fixture library unavailable")
    model_path = real_root / "models/api-fixture/model.gguf"
    if not binary.exists() or not model_path.exists():
        pytest.skip("Native fixture unavailable")
    tokenizer = ByteBPE.from_json(real_root / "models/api-fixture/tokenizer.json")
    with NativeModel(model_path, library=binary) as model:
        model.attach_tokenizer(tokenizer)
        identity = tokenizer_identity(tokenizer, 256)
        session = NativeSpeculativeModel(model, [1, 2], identity)
        try:
            assert np.isclose(session.distribution().sum(), 1)
        finally:
            session.close()
        pieces = list(identity.pieces)
        pieces[0], pieces[1] = pieces[1], pieces[0]
        with pytest.raises(ValueError, match="token bytes"):
            NativeSpeculativeModel(
                model, [1, 2], TokenizerIdentity(tuple(pieces), identity.configuration, 256)
            )
