from pathlib import Path
import pytest
from tvara.reference.oracle import load_oracle, oracle_logits

ROOT = Path(__file__).resolve().parents[2]


def test_pinned_oracle_is_cpu_float32_and_repeatable():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not (ROOT / "models/tiny-llama/model.safetensors").exists():
        pytest.skip("Pinned optional checkpoint has not been downloaded")
    model = load_oracle(ROOT / "models/tiny-llama", ROOT / "tests/fixtures/reference-model.json")
    assert str(next(model.parameters()).device) == "cpu"
    first = oracle_logits(model, [1, 2, 3])
    second = oracle_logits(model, [1, 2, 3])
    assert (first == second).all()
    assert first.shape == (3, 32000)
