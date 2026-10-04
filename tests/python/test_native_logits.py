from pathlib import Path
import subprocess
import numpy as np
import pytest
from tvara.reference.metrics import compare_logits

ROOT = Path(__file__).resolve().parents[2]


def test_native_executor_matches_full_oracle_logits(tmp_path):
    binary = ROOT / "build/debug/tvara_model_check.exe"
    if not binary.exists():
        binary = ROOT / "build/debug/tvara_model_check"
    model = ROOT / "models/tiny-llama/reference.gguf"
    if not binary.exists() or not model.exists():
        pytest.skip("Native binary or optional checkpoint unavailable")
    with np.load(ROOT / "tests/fixtures/reference-logits.npz") as fixture:
        for i in range(3):
            output = tmp_path / f"logits-{i}.bin"
            subprocess.run(
                [
                    str(binary),
                    str(model),
                    str(output),
                    *(str(int(t)) for t in fixture[f"tokens_{i}"]),
                ],
                check=True,
                capture_output=True,
            )
            expected = fixture[f"logits_{i}"]
            actual = np.fromfile(output, dtype=np.float32).reshape(expected.shape)
            report = compare_logits(actual, expected)
            assert report["within_tolerance"] and report["top1_agreement"] == 1, report
