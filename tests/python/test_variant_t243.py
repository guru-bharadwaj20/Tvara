import pytest
import hashlib
import json
from pathlib import Path

tokenizers = pytest.importorskip("tokenizers")
from tvara.qwen import qwen_tokenizer  # noqa: E402


def test_pinned_qwen_special_tokens():
    root = Path(__file__).resolve().parents[2]
    path = root / "models/qwen-tokenizer/tokenizer.json"
    if not path.exists():
        pytest.skip(
            "Optional pinned tokenizer artifact unavailable; run tools/fetch_test_artifacts.py"
        )
    manifest = json.loads((root / "tests/fixtures/tokenizer-model.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    ours, ids = qwen_tokenizer(path)
    reference = tokenizers.Tokenizer.from_file(str(path))
    for text in (
        "<|im_start|>user\nHello भारत 😀12345<|im_end|>",
        "<|endoftext|>",
        "  a\n\n123456789",
        "e\u0301 café",
    ):
        assert (
            ours.encode(text, allow_special=True)
            == reference.encode(text, add_special_tokens=False).ids
        )
    for name, ident in ids.items():
        assert ours.encode(name, allow_special=True) == [ident]
        assert ours.decode([ident]) == name
