"""Vocabulary masks over immutable grammar states; one constraint per generation."""

import codecs

from collections import OrderedDict

from dataclasses import dataclass, field
from .grammar import GrammarState  # noqa: F401


@dataclass
class TrieNode:
    children: dict = field(default_factory=dict)
    tokens: list = field(default_factory=list)


class VocabularyTrie:
    def __init__(self, vocabulary):
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary or any(not isinstance(raw, bytes) for raw in self.vocabulary):
            raise ValueError("Vocabulary must contain byte strings")
        self.root = TrieNode()
        self.node_count = 1
        for token, raw in enumerate(self.vocabulary):
            if not raw:
                continue
            node = self.root
            for byte in raw:
                if byte not in node.children:
                    node.children[byte] = TrieNode()
                    self.node_count += 1
                node = node.children[byte]
            node.tokens.append(token)


@dataclass(frozen=True)
class ByteState:
    grammar_state: GrammarState
    pending: bytes = b""


def advance_byte(grammar, state, byte):
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    decoder.setstate((state.pending, 0))
    try:
        text = decoder.decode(bytes([byte]), final=False)
    except UnicodeDecodeError:
        return None
    next_state = state.grammar_state
    for char in text:
        next_state = next_state.advance(grammar, char)
        if next_state is None:
            return None
    pending = decoder.getstate()[0]
    if pending and not pending_can_match(grammar, next_state, pending):
        return None
    return ByteState(next_state, pending)


def trie_allowed(trie, grammar, state, stats=None):
    result = []
    pending = [(trie.root, state)]
    while pending:
        node, current = pending.pop()
        if stats is not None:
            stats["visited"] = stats.get("visited", 0) + 1
        result.extend(node.tokens)
        for byte, child in node.children.items():
            candidate = advance_byte(grammar, current, byte)
            if candidate is not None:
                pending.append((child, candidate))
            elif stats is not None:
                stats["pruned"] = stats.get("pruned", 0) + 1
    return tuple(sorted(result))


@dataclass(frozen=True)
class TokenMask:
    size: int
    bits: int

    @classmethod
    def from_ids(cls, size, ids):
        if not isinstance(size, int) or size < 1:
            raise ValueError("Invalid mask size")
        bits = 0
        for token in ids:
            if not isinstance(token, int) or not 0 <= token < size:
                raise ValueError("Invalid mask token")
            bits |= 1 << token
        return cls(size, bits)

    def allows(self, token):
        return 0 <= token < self.size and bool(self.bits & (1 << token))

    def ids(self):
        return tuple(i for i in range(self.size) if self.allows(i))

    def packed(self):
        return self.bits.to_bytes((self.size + 7) // 8, "little")


def state_kind(grammar, state):
    from .grammar import closure, CharSet

    stacks = closure(grammar, state.grammar_state.stacks)
    if state.pending:
        return "utf8-dependent"
    if all(all(isinstance(symbol, CharSet) for symbol in stack) for stack in stacks):
        return "terminal-only"
    return "stack-dependent"


class MaskEngine:
    def __init__(self, vocabulary, grammar, eos_id=None, max_entries=1024):
        self.trie = VocabularyTrie(vocabulary)
        self.grammar = grammar
        self.eos_id = eos_id
        if eos_id is not None and not 0 <= eos_id < len(vocabulary):
            raise ValueError("Invalid EOS ID")
        if not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("Invalid mask cache bound")
        self.max_entries = max_entries
        self.precomputed = {}
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def compute(self, state):
        ids = list(trie_allowed(self.trie, self.grammar, state))
        if self.eos_id is not None:
            ids = [token for token in ids if token != self.eos_id]
            if not state.pending and state.grammar_state.accepting(self.grammar):
                ids.append(self.eos_id)
        return TokenMask.from_ids(len(self.trie.vocabulary), ids)

    def precompute(self, states):
        for state in states:
            if len(self.precomputed) >= self.max_entries and state not in self.precomputed:
                raise ValueError("Precompute limit reached")
            self.precomputed[state] = self.compute(state)

    def mask(self, state):
        if state in self.precomputed:
            self.hits += 1
            return self.precomputed[state]
        if state in self.cache:
            self.hits += 1
            self.cache.move_to_end(state)
            return self.cache[state]
        self.misses += 1
        result = self.compute(state)
        self.cache[state] = result
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
        return result


def naive_allowed(vocabulary, grammar, state):
    result = []
    for token, raw in enumerate(vocabulary):
        if not raw:
            continue
        candidate = state
        for byte in raw:
            candidate = advance_byte(grammar, candidate, byte)
            if candidate is None:
                break
        if candidate is not None:
            result.append(token)
    return tuple(result)


def pending_can_match(grammar, state, pending):
    from .grammar import closure

    first = pending[0]
    length = 2 if first < 0xE0 else 3 if first < 0xF0 else 4
    low = bytearray(pending)
    high = bytearray(pending)
    for position in range(len(pending), length):
        minimum = 0x80
        maximum = 0xBF
        if position == 1:
            if first == 0xE0:
                minimum = 0xA0
            elif first == 0xED:
                maximum = 0x9F
            elif first == 0xF0:
                minimum = 0x90
            elif first == 0xF4:
                maximum = 0x8F
        low.append(minimum)
        high.append(maximum)
    left = ord(low.decode("utf8"))
    right = ord(high.decode("utf8"))
    for stack in closure(grammar, state.stacks):
        if not stack:
            continue
        charset = stack[0]
        if not charset.negate:
            if any(a <= right and b >= left for a, b in charset.ranges):
                return True
        else:
            uncovered = left
            for a, b in sorted(charset.ranges):
                if b < uncovered:
                    continue
                if a > uncovered:
                    break
                uncovered = max(uncovered, b + 1)
            if uncovered <= right:
                return True
    return False


class TokenConstraint:
    def __init__(self, engine):
        self.engine = engine
        self.state = ByteState(GrammarState.start(engine.grammar))
        self.ended = False

    def mask_logits(self, logits):
        import numpy as np

        values = np.asarray(logits, dtype=float)
        if values.ndim != 1 or len(values) != len(self.engine.trie.vocabulary):
            raise ValueError("Logit vocabulary mismatch")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite model logits")
        ids = self.engine.mask(self.state).ids()
        if not ids:
            raise ValueError("No grammar-valid tokens remain")
        result = np.full(values.shape, -np.inf)
        result[list(ids)] = values[list(ids)]
        return result

    def accept(self, token):
        if self.ended or not self.engine.mask(self.state).allows(token):
            raise ValueError("Invalid constrained token")
        if token == self.engine.eos_id:
            self.ended = True
            return
        state = self.state
        for byte in self.engine.trie.vocabulary[token]:
            state = advance_byte(self.engine.grammar, state, byte)
        self.state = state

    @property
    def complete(self):
        return not self.state.pending and self.state.grammar_state.accepting(self.engine.grammar)


def tokenizer_bytes(tokenizer):
    return tuple(
        b"" if token in tokenizer.special_ids else tokenizer.token_bytes(token)
        for token in range(len(tokenizer.vocabulary))
    )
