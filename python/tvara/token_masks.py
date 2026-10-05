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
