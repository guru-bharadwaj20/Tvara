import hashlib
import json
from pathlib import Path
import random
import pytest
from tvara.tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parents[2]


def test_qwen_token_ids_match_independent_rust_tokenizer():
    rust = pytest.importorskip("tokenizers")
    path = ROOT / "models/qwen-tokenizer/tokenizer.json"
    if not path.exists():
        pytest.skip("Optional pinned tokenizer artifact unavailable")
    manifest = json.loads((ROOT / "tests/fixtures/tokenizer-model.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    ours = ByteBPE.from_json(path)
    reference = rust.Tokenizer.from_file(str(path))
    rng = random.Random(2026)
    alphabet = list("Hello world!12345\n\t éभारतதமிழ்😀🚀")
    prompts = [
        "",
        "Hello world",
        "123456789",
        "a\r\nb",
        "e\u0301",
        "<|im_start|>user\nhello<|im_end|>",
    ]
    prompts += ["".join(rng.choices(alphabet, k=rng.randrange(1, 80))) for _ in range(1000)]
    for text in prompts:
        assert (
            ours.encode(text, allow_special=True)
            == reference.encode(text, add_special_tokens=False).ids
        ), repr(text)
