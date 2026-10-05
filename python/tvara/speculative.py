"""Exact speculative sampling with explicit sequence ownership and rollback."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class TokenizerIdentity:
    pieces: tuple[bytes, ...]
    configuration: str
    eos_id: int | None = None

    def __post_init__(self):
        if (
            not isinstance(self.pieces, tuple)
            or not self.pieces
            or not isinstance(self.configuration, str)
            or not self.configuration
            or any(not isinstance(p, bytes) for p in self.pieces)
        ):
            raise ValueError("A complete tokenizer identity is required")
        if self.eos_id is not None and (
            not isinstance(self.eos_id, int) or not 0 <= self.eos_id < len(self.pieces)
        ):
            raise ValueError("Invalid EOS identity")


class SpeculativeModel(Protocol):
    identity: TokenizerIdentity
    history: list[int]

    def distribution(self) -> np.ndarray: ...
    def advance(self, token: int) -> None: ...
    def checkpoint(self) -> int: ...
    def rollback(self, checkpoint: int) -> None: ...


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid probability distribution")
    return p / p.sum()


def choose(p, rng):
    p = probabilities(p)
    return int(rng.choice(len(p), p=p))
