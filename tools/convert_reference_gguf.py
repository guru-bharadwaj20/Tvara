"""Export the tiny arithmetic fixture, not a production tokenizer-equipped model."""

import json
from pathlib import Path
import struct
import numpy as np
from safetensors.numpy import load_file


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def convert(source="models/tiny-llama", output="models/tiny-llama/reference.gguf"):
    source = Path(source)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    original = load_file(str(source / "model.safetensors"))
    names = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    for i in range(config["num_hidden_layers"]):
        names.update(
            {f"model.layers.{i}." + key: f"blk.{i}." + value for key, value in suffixes.items()}
        )
    tensors = {
        names[name]: np.asarray(array, dtype="<f4")
        for name, array in original.items()
        if name in names
    }
    metadata = [
        ("general.architecture", 8, string("llama")),
        ("tvara.rope.layout", 8, string("split")),
    ]
    for key, value in {
        "embedding_length": config["hidden_size"],
        "feed_forward_length": config["intermediate_size"],
        "block_count": config["num_hidden_layers"],
        "attention.head_count": config["num_attention_heads"],
        "attention.head_count_kv": config["num_key_value_heads"],
        "context_length": config["max_position_embeddings"],
    }.items():
        metadata.append(("llama." + key, 4, struct.pack("<I", value)))
    metadata += [
        ("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", config["rms_norm_eps"])),
        ("llama.rope.freq_base", 6, struct.pack("<f", config["rope_theta"])),
        (
            "tokenizer.ggml.tokens",
            9,
            struct.pack("<IQ", 8, config["vocab_size"])
            + b"".join(string(f"fixture-{i}") for i in range(config["vocab_size"])),
        ),
    ]
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
    header += b"".join(string(key) + struct.pack("<I", kind) + data for key, kind, data in metadata)
    payload = bytearray()
    for name, array in tensors.items():
        payload.extend(b"\x00" * ((-len(payload)) % 32))
        offset = len(payload)
        header += (
            string(name)
            + struct.pack("<I", array.ndim)
            + b"".join(struct.pack("<Q", d) for d in array.shape[::-1])
            + struct.pack("<IQ", 0, offset)
        )
        payload.extend(array.tobytes())
    header += b"\x00" * ((-len(header)) % 32)
    Path(output).write_bytes(header + payload)
    return output


if __name__ == "__main__":
    print(convert())
