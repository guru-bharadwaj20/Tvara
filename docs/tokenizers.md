# Tokenizer coverage

The Python tokenizer implements reversible byte-level BPE with ranked merges and
Unicode-aware regex pretokenization. It reads supported tokenizer.json BPE files,
including Qwen's NFC normalizer and explicit regex split followed by byte mapping.
Special strings are recognized only with allow_special=True; raw encoding does
not insert BOS/EOS. Prompt insertion follows validated metadata policy separately.

The Qwen2.5-0.5B-Instruct tokenizer artifact is local and pinned by checksum. Tests
compare exact token IDs with the independent Rust tokenizers library on 1,000
seeded Unicode strings and fixed special-token/normalization cases. A missing local
artifact causes an explicit optional-test skip. Weights are not downloaded for
these tokenizer checks.

SentencePiece support accepts scored GGUF vocabularies and implements a unigram
Viterbi lattice with explicit NFKC/NFC/no-normalization and whitespace options.
It includes unknown and byte fallback plus joined-byte UTF-8 decoding. This does
not implement arbitrary binary .model normalization maps or SentencePiece BPE
training-model behavior. That compatibility remains unverified.

Added-token lstrip/rstrip/single-word rules, BPE dropout, and arbitrary normalizer
pipelines are rejected. Byte-level streaming preserves incomplete UTF-8 prefixes
until subsequent tokens arrive; final truncated bytes use the requested decoder
error policy. SentencePiece prefix-space removal is applied to complete decode,
not advertised as implemented for incremental streaming yet.
