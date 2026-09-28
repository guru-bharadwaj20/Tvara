"""Independent tokenizer implementations with explicit supported normalizers."""

import json
from pathlib import Path
import unicodedata

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


def byte_mapping():
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    forward = {b: chr(c) for b, c in zip(byte_values, codepoints)}
    return forward, {char: byte for byte, char in forward.items()}


def merge_bpe(symbols, ranks):
    symbols = list(symbols)
    while len(symbols) > 1:
        ranked = [
            (ranks.get((symbols[i], symbols[i + 1]), float("inf")), i)
            for i in range(len(symbols) - 1)
        ]
        rank, index = min(ranked)
        if rank == float("inf"):
            break
        pair = (symbols[index], symbols[index + 1])
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class ByteBPE(Tokenizer):
    def __init__(
        self,
        vocabulary,
        merges,
        pattern=GPT2_PATTERN,
        normalizer=None,
        special_ids=(),
        add_prefix_space=False,
    ):
        super().__init__(vocabulary, special_ids)
        import regex

        self.pattern = regex.compile(pattern)
        self.normalizer = normalizer
        if normalizer not in (None, "NFC", "NFKC"):
            raise ValueError("Unsupported BPE normalizer")
        self.forward_bytes, self.reverse_bytes = byte_mapping()
        self.ranks = {}
        for rank, merge in enumerate(merges):
            pair = tuple(merge.split(" ") if isinstance(merge, str) else merge)
            if (
                len(pair) != 2
                or pair in self.ranks
                or any(p not in self.ids for p in pair)
                or "".join(pair) not in self.ids
            ):
                raise ValueError("Malformed BPE merge table")
            self.ranks[pair] = rank
        if any(char not in self.ids for char in self.forward_bytes.values()):
            raise ValueError("Byte BPE requires all 256 base byte tokens")
        self.add_prefix_space = add_prefix_space

    def encode(self, text, allow_special=False):
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        if self.normalizer:
            text = unicodedata.normalize(self.normalizer, text)
        if self.add_prefix_space and text and not text.startswith(" "):
            text = " " + text
        result = []
        for match in self.pattern.finditer(text):
            symbols = "".join(self.forward_bytes[b] for b in match.group().encode("utf-8"))
            result.extend(self.ids[symbol] for symbol in merge_bpe(symbols, self.ranks))
        return result

    def token_bytes(self, token):
        word = self.vocabulary[token]
        if token in self.special_ids:
            return word.encode("utf-8")
        try:
            return bytes(self.reverse_bytes[c] for c in word)
        except KeyError as error:
            raise ValueError("Token is outside byte-mapped vocabulary") from error

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        model = d["model"]
        if (
            model["type"] != "BPE"
            or model.get("dropout") is not None
            or model.get("byte_fallback", False)
        ):
            raise ValueError("Unsupported tokenizer BPE variant")
        vocab = dict(model["vocab"])
        specials = []
        for item in d.get("added_tokens", []):
            if item.get("lstrip") or item.get("rstrip") or item.get("single_word"):
                raise ValueError("Added-token whitespace/word matching is unsupported")
            vocab[item["content"]] = item["id"]
            if item.get("special"):
                specials.append(item["id"])
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("Sparse tokenizer IDs")
        words = [None] * len(vocab)
        for word, index in vocab.items():
            words[index] = word
        pre = d["pre_tokenizer"]
        steps = pre["pretokenizers"] if pre["type"] == "Sequence" else [pre]
        pattern = GPT2_PATTERN
        add_prefix = False
        for step in steps:
            if step["type"] == "Split":
                pattern = step["pattern"]["Regex"]
            elif step["type"] == "ByteLevel":
                add_prefix = step.get("add_prefix_space", False)
            else:
                raise ValueError("Unsupported pre-tokenizer")
        normalizer = d.get("normalizer")
        return cls(
            words,
            model["merges"],
            pattern,
            normalizer["type"] if normalizer else None,
            specials,
            add_prefix,
        )


class SentencePiece(Tokenizer):
    def __init__(self, metadata, normalizer="NFKC", remove_extra_whitespace=True):
        if metadata.model not in ("llama", "t5", "unigram"):
            raise ValueError("Unsupported SentencePiece model")
        super().__init__(
            metadata.tokens, [i for i, t in enumerate(metadata.types) if t in (2, 3, 4)]
        )
        self.metadata = metadata
        self.normalizer = normalizer
        self.remove_extra_whitespace = remove_extra_whitespace
        if normalizer not in (None, "NFC", "NFKC"):
            raise ValueError("Unsupported SentencePiece normalizer")
        self.pieces = [
            (token, i, metadata.scores[i])
            for i, token in enumerate(metadata.tokens)
            if metadata.types[i] == 1
        ]
        self.unknown = next((i for i, t in enumerate(metadata.types) if t == 2), None)
        self.byte_ids = {
            int(token[3:5], 16): i
            for i, token in enumerate(metadata.tokens)
            if metadata.types[i] == 6
            and len(token) == 6
            and token.startswith("<0x")
            and token.endswith(">")
        }

    def encode(self, text, allow_special=False):
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        if self.normalizer:
            text = unicodedata.normalize(self.normalizer, text)
        if self.remove_extra_whitespace:
            text = " ".join(text.split())
        if not text:
            return []
        if self.metadata.add_space_prefix:
            text = " " + text
        text = text.replace(" ", chr(0x2581))
        best = [-float("inf")] * (len(text) + 1)
        paths = [None] * (len(text) + 1)
        best[0] = 0.0
        paths[0] = []
        for start in range(len(text)):
            if paths[start] is None:
                continue
            choices = [
                (len(piece), [index], score)
                for piece, index, score in self.pieces
                if text.startswith(piece, start)
            ]
            if not any(length == 1 for length, _, _ in choices):
                char = text[start]
                raw = char.encode("utf-8")
                if all(b in self.byte_ids for b in raw):
                    fallback = [self.byte_ids[b] for b in raw]
                elif self.unknown is not None:
                    fallback = [self.unknown]
                else:
                    raise ValueError("Text has no SentencePiece or byte fallback")
                choices.append((1, fallback, -100.0))
            for length, ids, score in choices:
                end = start + length
                candidate = best[start] + score
                if candidate > best[end]:
                    best[end] = candidate
                    paths[end] = paths[start] + ids
        if paths[-1] is None:
            raise ValueError("SentencePiece lattice has no accepting path")
        return paths[-1]

    def token_bytes(self, token):
        word = self.vocabulary[token]
        if self.metadata.types[token] == 6:
            if not (len(word) == 6 and word.startswith("<0x") and word.endswith(">")):
                raise ValueError("Malformed byte-fallback token")
            return bytes([int(word[3:5], 16)])
        return word.replace(chr(0x2581), " ").encode("utf-8")

    def decode(self, tokens, skip_special=False, errors="replace"):
        text = super().decode(tokens, skip_special, errors)
        if self.metadata.add_space_prefix and text.startswith(" "):
            text = text[1:]
        return text
