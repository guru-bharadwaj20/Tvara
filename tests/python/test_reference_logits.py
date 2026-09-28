from pathlib import Path
import numpy as np
import pytest
from tvara.reference.model import ReferenceModel
from tvara.reference.metrics import compare_logits

ROOT = Path(__file__).resolve().parents[2]


def test_reference_model_matches_pinned_hugging_face_logits():
    pytest.importorskip("safetensors")
    if not (ROOT / "models/tiny-llama/model.safetensors").exists():
        pytest.skip("Optional pinned checkpoint is unavailable")
    model = ReferenceModel.from_directory(ROOT / "models/tiny-llama")
    with np.load(ROOT / "tests/fixtures/reference-logits.npz") as fixture:
        for i in range(3):
            report = compare_logits(model.forward(fixture[f"tokens_{i}"]), fixture[f"logits_{i}"])
            assert report["within_tolerance"], report
            assert report["top1_agreement"] == 1, report
