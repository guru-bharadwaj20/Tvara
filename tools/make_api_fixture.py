"""Create a small random-weight native API fixture; this is not a trained model."""

import json
from pathlib import Path
import struct
import numpy as np
from tvara.tokenizer import byte_mapping


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def make_fixture(
    directory="models/api-fixture",
    benchmark=False,
    q8=False,
    layers=1,
    hidden=32,
    ffn=64,
    heads=4,
    kv_heads=2,
    context=64,
    output_std=0.02,
):
    """Defaults reproduce the pinned fixture bytes. Tensors are drawn in a fixed order
    (embeddings, output, then each block), so a model with fewer layers and the same
    seed shares its embeddings, output projection and leading blocks with a deeper one."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    forward, _ = byte_mapping()
    words = [forward[i] for i in range(256)] + ["<|endoftext|>"]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "vocab": {word: i for i, word in enumerate(words)},
            "merges": [],
            "dropout": None,
            "byte_fallback": False,
        },
        "added_tokens": [
            {
                "id": 256,
                "content": words[-1],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
    }
    (directory / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    rng = np.random.default_rng(2026)
    kv = hidden // heads * kv_heads
    shapes = {
        "token_embd.weight": (257, hidden),
        "output.weight": (257, hidden),
        "output_norm.weight": (hidden,),
    }
    for layer in range(layers):
        block = f"blk.{layer}."
        shapes |= {
            block + "attn_norm.weight": (hidden,),
            block + "ffn_norm.weight": (hidden,),
            block + "attn_q.weight": (hidden, hidden),
            block + "attn_k.weight": (kv, hidden),
            block + "attn_v.weight": (kv, hidden),
            block + "attn_output.weight": (hidden, hidden),
            block + "ffn_gate.weight": (ffn, hidden),
            block + "ffn_up.weight": (ffn, hidden),
            block + "ffn_down.weight": (hidden, ffn),
        }
    tensors = {
        name: np.ones(shape, dtype="<f4")
        if "norm.weight" in name
        else rng.normal(0, output_std if name == "output.weight" else 0.02, shape).astype("<f4")
        for name, shape in shapes.items()
    }
    metadata = [
        ("general.architecture", 8, string("llama")),
        ("tvara.rope.layout", 8, string("interleaved" if benchmark else "split")),
        ("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-6)),
        ("tokenizer.ggml.pre", 8, string("gpt-2")),
        ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 0)),
        ("tokenizer.ggml.model", 8, string("gpt2")),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 256)),
        (
            "tokenizer.ggml.tokens",
            9,
            struct.pack("<IQ", 8, len(words)) + b"".join(string(word) for word in words),
        ),
    ]
    for name, value in {
        "embedding_length": hidden,
        "feed_forward_length": ffn,
        "block_count": layers,
        "attention.head_count": heads,
        "attention.head_count_kv": kv_heads,
        "context_length": context,
    }.items():
        metadata.append(("llama." + name, 4, struct.pack("<I", value)))
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
    header += b"".join(
        string(name) + struct.pack("<I", kind) + data for name, kind, data in metadata
    )
    payload = bytearray()
    for name, array in tensors.items():
        kind = 8 if q8 and array.ndim == 2 else 0
        raw = array.tobytes()
        if kind == 8:
            blocks = array.reshape(-1, 32)
            payload_blocks = []
            for block in blocks:
                scale = float(np.max(np.abs(block))) / 127
                quant = (
                    np.zeros(32, dtype=np.int8)
                    if scale == 0
                    else np.clip(np.round(block / scale), -127, 127).astype(np.int8)
                )
                payload_blocks.append(np.asarray(scale, dtype="<f2").tobytes() + quant.tobytes())
            raw = b"".join(payload_blocks)
        payload.extend(b"\x00" * ((-len(payload)) % 32))
        offset = len(payload)
        header += (
            string(name)
            + struct.pack("<I", array.ndim)
            + b"".join(struct.pack("<Q", d) for d in array.shape[::-1])
            + struct.pack("<IQ", kind, offset)
        )
        payload.extend(raw)
    header += b"\x00" * ((-len(header)) % 32)
    (directory / "model.gguf").write_bytes(header + payload)
    return directory


if __name__ == "__main__":
    print(make_fixture())
