from pathlib import Path
import pytest
from tvara.native import NativeModel

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native():
    binary = ROOT / "build/debug/tvara_capi.dll"
    if not binary.exists():
        binary = ROOT / "build/debug/libtvara_capi.so"
    model = ROOT / "models/tiny-llama/reference.gguf"
    if not binary.exists() or not model.exists():
        pytest.skip("Optional native model fixture unavailable")
    with NativeModel(model, library=binary) as handle:
        yield handle


def test_native_model_loads_cpu_weights_and_closes_idempotently(native):
    assert native.vocab_size == 32000 and native.context_size == 2048
    native.close()
    native.close()
    with pytest.raises(RuntimeError):
        native._check_open()


def test_metadata_and_explicit_tokenizer_attachment(native):
    from tvara.tokenizer import Tokenizer

    metadata = native.metadata
    assert metadata["general.architecture"] == "llama"

    class Catalog(Tokenizer):
        def encode(self, text, allow_special=False):
            return [self.ids[t] for t in text.split()]

        def token_bytes(self, token):
            return self.vocabulary[token].encode()

    native.attach_tokenizer(Catalog(metadata["tokenizer.ggml.tokens"]))
    assert native.encode("fixture-2 fixture-7") == [2, 7]
    assert native.decode_text([2]) == "fixture-2"
