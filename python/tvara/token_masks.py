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
