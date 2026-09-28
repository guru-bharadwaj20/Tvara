"""Independent tokenizer implementations with explicit supported normalizers."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TokenizerMetadata:
    model: str
    tokens: tuple[str, ...]
    scores: tuple[float, ...]
    types: tuple[int, ...]
    merges: tuple[str, ...] = ()
    bos_id: int | None = None
    eos_id: int | None = None
    add_bos: bool = False
    add_eos: bool = False
    add_space_prefix: bool = True

    @classmethod
    def from_mapping(cls, metadata):
        prefix = "tokenizer.ggml."
        tokens = tuple(metadata.get(prefix + "tokens", ()))
        if (
            not tokens
            or any(not isinstance(t, str) for t in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            raise ValueError("Tokenizer vocabulary must contain unique strings")
        scores = tuple(float(x) for x in metadata.get(prefix + "scores", [0.0] * len(tokens)))
        types = tuple(metadata.get(prefix + "token_type", [1] * len(tokens)))
        if len(scores) != len(tokens) or len(types) != len(tokens) or not np.isfinite(scores).all():
            raise ValueError("Malformed tokenizer scores or types")
        if any(t not in range(1, 7) for t in types):
            raise ValueError("Unknown tokenizer token type")

        def token_id(key):
            value = metadata.get(prefix + key)
            if value is not None and (not isinstance(value, int) or not 0 <= value < len(tokens)):
                raise ValueError("Special token ID outside vocabulary")
            return value

        return cls(
            metadata.get(prefix + "model", "llama"),
            tokens,
            scores,
            types,
            tuple(metadata.get(prefix + "merges", ())),
            token_id("bos_token_id"),
            token_id("eos_token_id"),
            bool(metadata.get(prefix + "add_bos_token", False)),
            bool(metadata.get(prefix + "add_eos_token", False)),
            bool(metadata.get(prefix + "add_space_prefix", True)),
        )
