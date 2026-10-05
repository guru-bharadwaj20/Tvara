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
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any() or p.max() <= 0:
        raise ValueError("Invalid probability distribution")
    p = p / p.max()
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


def residual_distribution(p, q):
    p, q = probabilities(p), probabilities(q)
    if p.shape != q.shape:
        raise ValueError("Residual distributions have different vocabularies")
    residual = np.maximum(p - q, 0)
    if not np.any(residual > 0):
        raise ValueError("Residual sampling is undefined without rejection mass")
    return probabilities(residual)


def accepted_bonus(target_rows, rng):
    if not target_rows:
        raise ValueError("Verification requires a bonus distribution")
    return choose(target_rows[-1], rng)


class NativeSpeculativeModel:
    """Owned sequence adapter; caller owns the NativeModel lifetime."""

    def __init__(self, model, prompt, identity, temperature=1.0):
        import ctypes

        self.model, self.identity = model, identity
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Native speculative temperature must be positive")
        self.temperature = float(temperature)
        self._ctypes = ctypes
        if len(identity.pieces) != model.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match native model")
        self.history = list(prompt)
        if not self.history or any(
            not isinstance(t, int) or not 0 <= t < model.vocab_size for t in self.history
        ):
            raise ValueError("Invalid speculative prompt")
        lib = model._library
        lib.tvara_sequence_decode_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.tvara_sequence_decode_batch.restype = ctypes.c_int
        lib.tvara_sequence_truncate.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.tvara_sequence_truncate.restype = ctypes.c_int
        self.sequence = model._new_sequence()
        try:
            for token in self.history:
                self._logits = model.logits(self.sequence, token)
        except BaseException:
            self.close()
            raise

    def distribution(self):
        values = self._logits.astype(np.float64) / self.temperature
        if not np.isfinite(values).all():
            raise ValueError("Native logits are nonfinite")
        return probabilities(np.exp(values - values.max()))

    def advance(self, token):
        self._logits = self.model.logits(self.sequence, token)
        self.history.append(token)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, checkpoint):
        if not isinstance(checkpoint, int) or not 1 <= checkpoint <= len(self.history):
            raise ValueError("Invalid native checkpoint")
        if checkpoint == len(self.history):
            return
        with self.model._lock:
            status = self.model._library.tvara_sequence_truncate(
                self.model._handle, self.sequence, checkpoint - 1
            )
            if status:
                self.model._raise_error()
            del self.history[checkpoint:]
            self._logits = self.model.logits(self.sequence, self.history[-1])

    def verify_batch(self, proposals):
        old = self.distribution()
        proposed_tokens = [p.token for p in proposals]
        if any(
            not isinstance(t, int) or not 0 <= t < self.model.vocab_size for t in proposed_tokens
        ):
            raise ValueError("Invalid native proposal token")
        tokens = np.asarray(proposed_tokens, dtype=np.uint32)
        output = np.empty((len(tokens), self.model.vocab_size), dtype=np.float32)
        ctypes = self._ctypes
        with self.model._lock:
            status = self.model._library.tvara_sequence_decode_batch(
                self.model._handle,
                self.sequence,
                tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                len(tokens),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.size,
            )
            if status:
                self.model._raise_error()
        rows = [old]
        for token, logits in zip(tokens, output):
            self.history.append(int(token))
            self._logits = logits.copy()
            rows.append(self.distribution())
        return rows

    def close(self):
        if getattr(self, "sequence", None) is not None:
            self.model._drop_sequence(self.sequence)
            self.sequence = None


def rollback_target(target, checkpoint, accepted):
    if not isinstance(accepted, int) or accepted < 0:
        raise ValueError("Invalid accepted prefix")
    target.rollback(checkpoint + accepted)


@dataclass(frozen=True)
class StepResult:
    tokens: tuple[int, ...]
    proposed: int
    accepted: int


def speculative_step(draft, target, length, rng, constraint=None):
    require_compatible(draft, target)
    if constraint is not None:
        require_grammar_identity(target.identity, constraint)
    if draft.history != target.history:
        raise ValueError("Draft and target histories differ")
    d_checkpoint, t_checkpoint = draft.checkpoint(), target.checkpoint()
    original_constraint = clone_constraint(constraint) if constraint is not None else None
    draft_constraint = clone_constraint(constraint) if constraint is not None else None
    target_constraint = clone_constraint(constraint) if constraint is not None else None
    try:
        proposals = propose(draft, length, rng, draft_constraint)
        rows = verify(target, proposals, target_constraint)
        accepted = 0
        for index, proposal in enumerate(proposals):
            if rng.random() >= acceptance_probability(
                rows[index], proposal.distribution, proposal.token
            ):
                break
            accepted += 1
        output = [p.token for p in proposals[:accepted]]
        ended = output and (
            output[-1] == target.identity.eos_id
            or (
                target_constraint is not None
                and target_constraint.ended
                and accepted == len(proposals)
            )
        )
        if not ended:
            token = (
                accepted_bonus(rows, rng)
                if accepted == len(proposals)
                else choose(
                    residual_distribution(rows[accepted], proposals[accepted].distribution), rng
                )
            )
            output.append(token)
        rollback_target(target, t_checkpoint, accepted)
        draft.rollback(d_checkpoint + accepted)
        for token in output[accepted:]:
            target.advance(token)
            draft.advance(token)
        if constraint is not None:
            for token in output:
                constraint.accept(token)
        return StepResult(tuple(output), len(proposals), accepted)
    except BaseException:
        target.rollback(t_checkpoint)
        draft.rollback(d_checkpoint)
        if constraint is not None:
            constraint.state = original_constraint.state
            constraint.ended = original_constraint.ended
        raise


def grammar_checkpoint(constraint):
    return (constraint.state, constraint.ended)


def require_grammar_identity(identity, constraint):
    inventory = getattr(constraint.engine, "trie", None)
    if inventory is not None and (
        tuple(inventory.vocabulary) != identity.pieces
        or constraint.engine.eos_id != identity.eos_id
    ):
        raise ValueError("Grammar token inventory does not match model token IDs")
