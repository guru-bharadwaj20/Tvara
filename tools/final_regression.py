"""Independent HF-fixture comparisons using the actual native decoder."""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from tvara.native import NativeModel
from tvara.reference.metrics import compare_logits, perplexity


def main():
    reports = []
    with (
        NativeModel("models/tiny-llama/reference.gguf") as model,
        np.load("tests/fixtures/reference-logits.npz") as fixture,
    ):
        for index in range(3):
            tokens = fixture[f"tokens_{index}"].tolist()
            sequence = model._new_sequence()
            try:
                actual = np.asarray([model.logits(sequence, token) for token in tokens])
            finally:
                model._drop_sequence(sequence)
            expected = fixture[f"logits_{index}"]
            comparison = compare_logits(actual, expected)
            if not comparison["within_tolerance"] or comparison["top1_agreement"] != 1:
                raise AssertionError(comparison)
            observed = perplexity(actual, tokens)
            oracle = perplexity(expected, tokens)
            if not np.isclose(observed, oracle, rtol=1e-5):
                raise AssertionError("Perplexity regression")
            reports.append(
                {
                    "tokens": tokens,
                    "comparison": comparison,
                    "native_perplexity": observed,
                    "oracle_perplexity": oracle,
                }
            )
        greedy = json.loads(Path("tests/fixtures/greedy.json").read_text(encoding="utf8"))
        generated = model.generate(greedy["prompt"], max_new_tokens=len(greedy["generated"]))
        if generated != greedy["generated"]:
            raise AssertionError("Greedy regression")
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "logits": reports,
        "greedy_matches": True,
        "scope": "Pinned random tiny HF Llama oracle fixtures; perplexity is a numerical regression metric, not trained-model language quality",
    }
    target = Path("bench/results/final-regression.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"prompts": len(reports), "greedy_matches": True}))


if __name__ == "__main__":
    main()
