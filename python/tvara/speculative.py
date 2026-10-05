"""Exact speculative sampling with explicit sequence ownership and rollback."""

from dataclasses import dataclass
from typing import Protocol
import hashlib
import json
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


def tokenizer_fingerprint(identity):
    payload = {
        "pieces": [p.hex() for p in identity.pieces],
        "configuration": identity.configuration,
        "eos_id": identity.eos_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_compatible(draft, target):
    if tokenizer_fingerprint(draft.identity) != tokenizer_fingerprint(target.identity):
        raise ValueError("Draft and target tokenizers must have identical token IDs and rules")


def clone_constraint(constraint):
    candidate = type(constraint)(constraint.engine)
    candidate.state = constraint.state
    candidate.ended = constraint.ended
    return candidate


def constrained_probabilities(p, constraint):
    if constraint.ended:
        raise ValueError("Cannot propose after grammar EOS")
    allowed = constraint.engine.mask(constraint.state).ids()
    masked = np.zeros_like(p)
    masked[list(allowed)] = p[list(allowed)]
    return probabilities(masked)


@dataclass(frozen=True)
class Proposal:
    token: int
    distribution: np.ndarray


def propose(draft, length, rng, constraint=None):
    if not isinstance(length, int) or not 0 <= length <= 64:
        raise ValueError("Draft length must be between zero and 64")
    proposals = []
    for _ in range(length):
        q = probabilities(draft.distribution())
        if constraint is not None:
            q = constrained_probabilities(q, constraint)
        token = choose(q, rng)
        proposals.append(Proposal(token, q.copy()))
        draft.advance(token)
        if constraint is not None:
            constraint.accept(token)
            if constraint.ended:
                break
        if token == draft.identity.eos_id:
            break
    return proposals


def verify(target, proposals, constraint=None):
    if constraint is None and hasattr(target, "verify_batch"):
        return target.verify_batch(proposals)
    distributions = []
    for proposal in proposals:
        p = probabilities(target.distribution())
        if constraint is not None:
            p = constrained_probabilities(p, constraint)
        distributions.append(p)
        target.advance(proposal.token)
        if constraint is not None:
            constraint.accept(proposal.token)
    extra = probabilities(target.distribution())
    if constraint is not None and not constraint.ended:
        extra = constrained_probabilities(extra, constraint)
    distributions.append(extra)
    return distributions


def acceptance_probability(p, q, token):
    p, q = probabilities(p), probabilities(q)
    if p.shape != q.shape or not 0 <= token < len(p):
        raise ValueError("Incompatible acceptance distributions")
    if q[token] <= 0:
        raise ValueError("A proposed token must have positive draft probability")
    return min(1.0, float(p[token] / q[token]))
