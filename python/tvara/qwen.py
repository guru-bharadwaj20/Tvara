"""Qwen2 special-token inspection for the independent BPE tokenizer."""

from .tokenizer import ByteBPE


def qwen_tokenizer(path):
    tokenizer = ByteBPE.from_json(path)
    required = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
    ids = {name: tokenizer.ids.get(name) for name in required}
    if any(value is None or value not in tokenizer.special_ids for value in ids.values()):
        raise ValueError("Qwen2 chat special tokens are missing or not marked special")
    return tokenizer, ids
