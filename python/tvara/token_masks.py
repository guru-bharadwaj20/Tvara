"""Vocabulary masks over immutable grammar states; one constraint per generation."""

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
    if byte >= 128:
        return None
    next_state = state.grammar_state.advance(grammar, chr(byte))
    return ByteState(next_state) if next_state is not None else None


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
        return self.precomputed.get(state) or self.compute(state)
