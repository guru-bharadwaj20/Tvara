"""Compare native outputs to an offline NumPy random-weight Llama fixture."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np
from make_api_fixture import make_fixture
from tvara.reference.model import Config, ReferenceModel
from tvara.reference.metrics import compare_logits


def weights():
    # Same deterministic fixture data, independently evaluated NumPy mathematics.
    rng = np.random.default_rng(2026)
    shapes = {
        "model.embed_tokens.weight": (257, 32),
        "lm_head.weight": (257, 32),
        "model.norm.weight": (32,),
        "model.layers.0.input_layernorm.weight": (32,),
        "model.layers.0.post_attention_layernorm.weight": (32,),
        "model.layers.0.self_attn.q_proj.weight": (32, 32),
        "model.layers.0.self_attn.k_proj.weight": (16, 32),
        "model.layers.0.self_attn.v_proj.weight": (16, 32),
        "model.layers.0.self_attn.o_proj.weight": (32, 32),
        "model.layers.0.mlp.gate_proj.weight": (64, 32),
        "model.layers.0.mlp.up_proj.weight": (64, 32),
        "model.layers.0.mlp.down_proj.weight": (32, 64),
    }
    return {
        name: np.ones(shape, dtype=np.float32)
        if "norm.weight" in name
        else rng.normal(0, 0.02, shape).astype(np.float32)
        for name, shape in shapes.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    reference = ReferenceModel(Config(257, 32, 64, 1, 4, 2, max_position_embeddings=64), weights())
    records = []
    with tempfile.TemporaryDirectory(prefix="tvara-portability-") as temporary:
        directory = Path(temporary)
        make_fixture(directory)
        for paged in (False, True):
            for index, tokens in enumerate(([1], [1, 2, 3, 9], [7, 8, 9, 10, 11, 12, 13, 14, 15])):
                expected = reference.forward(tokens)
                output = directory / f"logits-{paged}-{index}.bin"
                subprocess.run(
                    [
                        str(binary),
                        str(directory / "model.gguf"),
                        str(output),
                        *(["--paged"] if paged else []),
                        *(str(t) for t in tokens),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                actual = np.fromfile(output, dtype=np.float32).reshape(expected.shape)
                report = compare_logits(actual, expected)
                assert report["within_tolerance"] and report["top1_agreement"] == 1, report
                records.append(dict(paged=paged, tokens=len(tokens), **report))
    result = dict(
        status="validated",
        machine=platform.machine(),
        scope="offline deterministic random-weight Llama, native contiguous and paged outputs against independent NumPy reference",
        records=records,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
