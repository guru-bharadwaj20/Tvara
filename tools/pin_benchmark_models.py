"""Generate checksum-pinned random-weight matched benchmark files."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from make_api_fixture import make_fixture


def main():
    records = []
    for quant in (False, True):
        directory = make_fixture(
            "models/benchmark-" + ("q8" if quant else "f32"), benchmark=True, q8=quant
        )
        model = directory / "model.gguf"
        records.append(
            {
                "path": str(model).replace("\\", "/"),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "bytes": model.stat().st_size,
                "quantization": "Q8_0 matrices/F32 norms" if quant else "F32",
                "context": 64,
                "vocabulary": 257,
                "seed": 2026,
            }
        )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Random-weight tiny Llama fixtures, not pretrained language quality or representative large-model speed",
        "models": records,
    }
    target = Path("bench/results/benchmark-models.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
