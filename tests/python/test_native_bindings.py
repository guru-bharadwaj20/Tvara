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
