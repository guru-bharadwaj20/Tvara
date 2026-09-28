import json
from pathlib import Path
import pytest
from tvara.reference.model import ReferenceModel
from tvara.reference.generation import greedy_generate

ROOT = Path(__file__).resolve().parents[2]


def test_greedy_tokens_match_independent_oracle():
    if not (ROOT / "models/tiny-llama/model.safetensors").exists():
        pytest.skip("Optional pinned checkpoint unavailable")
    model = ReferenceModel.from_directory(ROOT / "models/tiny-llama")
    fixture = json.loads((ROOT / "tests/fixtures/greedy.json").read_text())
    assert greedy_generate(model, fixture["prompt"], 6) == fixture["generated"]
    first = fixture["generated"][0]
    assert greedy_generate(model, fixture["prompt"], 6, eos_token_id=first) == [first]
