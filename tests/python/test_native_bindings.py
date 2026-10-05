from pathlib import Path
import pytest
from tvara.native import NativeModel, find_library

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native():
    try:
        binary = find_library()
    except FileNotFoundError:
        pytest.skip("Optional native library unavailable")
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


def test_native_generation_matches_greedy_oracle_and_releases_sequences(native):
    import json

    fixture = json.loads((ROOT / "tests/fixtures/greedy.json").read_text())
    assert native.generate(fixture["prompt"], max_new_tokens=6) == fixture["generated"]
    first = native.generate([1, 2], max_new_tokens=3, temperature=1, seed=42)
    assert first == native.generate([1, 2], max_new_tokens=3, temperature=1, seed=42)
    stream = native.iter_tokens([1, 2], max_new_tokens=4)
    next(stream)
    stream.close()
    assert native.generate([1, 2], max_new_tokens=1)


@pytest.mark.parametrize("temperature", [0, 1])
def test_native_grammar_mask_precedes_all_sampling_filters(native, temperature):
    from tvara.grammar import Grammar
    from tvara.token_masks import TokenConstraint, MaskEngine

    vocabulary = [b""] * native.vocab_size
    vocabulary[17] = b"a"
    constraint = TokenConstraint(
        MaskEngine(vocabulary, Grammar.from_text('root ::= "a"'), eos_id=2)
    )
    assert native.generate(
        [1, 2],
        max_new_tokens=2,
        temperature=temperature,
        top_k=1,
        min_p=0.9,
        top_p=0.1,
        eos_id=2,
        token_constraint=constraint,
    ) == [17, 2]
    assert constraint.complete and constraint.ended
