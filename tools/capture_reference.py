"""Capture pinned oracle logits; checkpoint stays in ignored local storage."""

import argparse
import json
from pathlib import Path
import numpy as np
from tvara.reference.oracle import load_oracle, oracle_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/tiny-llama")
    p.add_argument("--output", default="tests/fixtures/reference-logits.npz")
    args = p.parse_args()
    prompts = [[1, 2, 3, 4], [0, 17, 400, 2], [1, 42]]
    oracle = load_oracle(args.model, "tests/fixtures/reference-model.json")
    result = {f"tokens_{i}": np.asarray(tokens, dtype=np.int64) for i, tokens in enumerate(prompts)}
    result.update(
        {f"logits_{i}": oracle_logits(oracle, tokens) for i, tokens in enumerate(prompts)}
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(
        json.dumps(
            {
                "prompts": len(prompts),
                "precision": "float32",
                "device": "cpu",
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
