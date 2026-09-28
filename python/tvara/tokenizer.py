"""Independent tokenizer implementations with explicit supported normalizers."""

from abc import ABC, abstractmethod

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


class Tokenizer(ABC):
    def __init__(self, vocabulary, special_ids=()):
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary must be nonempty and unique")
        self.ids = {word: i for i, word in enumerate(self.vocabulary)}
        self.special_ids = frozenset(special_ids)
        if any(
            not isinstance(i, int) or not 0 <= i < len(self.vocabulary) for i in self.special_ids
        ):
            raise ValueError("Invalid special token IDs")

    @abstractmethod
    def encode(self, text, allow_special=False):
        pass

    @abstractmethod
    def token_bytes(self, token):
        pass

    def decode(self, tokens, skip_special=False, errors="replace"):
        output = []
        for token in tokens:
            if not isinstance(token, (int, np.integer)) or not 0 <= token < len(self.vocabulary):
                raise ValueError("Token ID outside vocabulary")
            if skip_special and token in self.special_ids:
                continue
            output.append(self.token_bytes(int(token)))
        return b"".join(output).decode("utf-8", errors=errors)
