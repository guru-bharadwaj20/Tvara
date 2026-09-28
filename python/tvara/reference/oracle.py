"""CPU-only Hugging Face oracle; optional dependencies are imported lazily."""

import hashlib
import json
from pathlib import Path


def load_oracle(directory, manifest=None):
    directory = Path(directory)
    if manifest is not None:
        specification = json.loads(Path(manifest).read_text(encoding="utf-8"))
        for name, expected in specification["sha256"].items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetensors.torch import load_file

    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = LlamaConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(
        load_file(str(directory / "model.safetensors"), device="cpu"), strict=True
    )
    return model.eval()


def oracle_logits(model, tokens):
    import torch

    with torch.inference_mode():
        ids = torch.tensor([tokens], dtype=torch.long, device="cpu")
        return model(ids, use_cache=False).logits[0].float().numpy().copy()
