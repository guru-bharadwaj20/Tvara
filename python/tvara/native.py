"""Owned ctypes handles for Tvara's CPU-only native model library."""

import ctypes
import json
import numpy as np
import os
from pathlib import Path
import threading
import weakref


def find_library(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    if os.environ.get("TVARA_LIBRARY"):
        return Path(os.environ["TVARA_LIBRARY"]).resolve()
    root = Path(__file__).resolve().parents[2]
    names = ["tvara_capi.dll", "libtvara_capi.dll", "libtvara_capi.so", "libtvara_capi.dylib"]
    for mode in ("release", "debug"):
        for name in names:
            candidate = root / "build" / mode / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Build tvara_capi with CMake or set TVARA_LIBRARY")


class NativeModel:
    def __init__(self, path, library=None, max_blocks=256, block_tokens=16):
        self._lock = threading.RLock()
        self._library = ctypes.CDLL(str(find_library(library)))
        lib = self._library
        lib.tvara_model_open.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.tvara_model_open.restype = ctypes.c_void_p
        lib.tvara_model_close.argtypes = [ctypes.c_void_p]
        lib.tvara_model_close.restype = None
        lib.tvara_model_vocab_size.argtypes = [ctypes.c_void_p]
        lib.tvara_model_vocab_size.restype = ctypes.c_size_t
        lib.tvara_model_context_size.argtypes = [ctypes.c_void_p]
        lib.tvara_model_context_size.restype = ctypes.c_size_t
        lib.tvara_last_error.argtypes = []
        lib.tvara_last_error.restype = ctypes.c_char_p
        if (
            not isinstance(max_blocks, int)
            or not isinstance(block_tokens, int)
            or max_blocks <= 0
            or block_tokens <= 0
        ):
            raise ValueError("Cache geometry must be positive integers")
        self._handle = lib.tvara_model_open(
            str(Path(path).resolve()).encode("utf-8"), max_blocks, block_tokens
        )
        if not self._handle:
            self._raise_error()
        self._finalizer = weakref.finalize(self, lib.tvara_model_close, self._handle)
        self.vocab_size = int(lib.tvara_model_vocab_size(self._handle))
        self.context_size = int(lib.tvara_model_context_size(self._handle))
        self._configure()

    def _configure(self):
        self._library.tvara_model_metadata.argtypes = [ctypes.c_void_p]
        self._library.tvara_model_metadata.restype = ctypes.c_char_p
        self._tokenizer = None
        lib = self._library
        lib.tvara_sequence_create.argtypes = [ctypes.c_void_p]
        lib.tvara_sequence_create.restype = ctypes.c_uint64
        lib.tvara_sequence_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.tvara_sequence_decode.restype = ctypes.c_int
        lib.tvara_sequence_drop.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.tvara_sequence_drop.restype = ctypes.c_int
        # CONFIGURE_BINDINGS
        pass

    def _raise_error(self):
        message = self._library.tvara_last_error()
        raise RuntimeError(
            message.decode("utf-8", errors="replace") if message else "Native operation failed"
        )

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("Model is closed")

    def close(self):
        with self._lock:
            if self._handle:
                self._finalizer()
                self._handle = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def metadata(self):
        with self._lock:
            self._check_open()
            raw = self._library.tvara_model_metadata(self._handle)
            if not raw:
                self._raise_error()
            return json.loads(raw.decode("utf-8"))

    def attach_tokenizer(self, tokenizer):
        words = tuple(tokenizer.vocabulary)
        if len(words) != self.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match model")
        with self._lock:
            metadata = self.metadata
            expected = metadata.get("tokenizer.ggml.tokens")
            if expected is not None and words != tuple(expected):
                raise ValueError("Tokenizer token IDs do not match the ordered model vocabulary")
            special_ids = set(getattr(tokenizer, "special_ids", ()))
            for name in ("bos", "eos", "unknown", "padding"):
                token = metadata.get(f"tokenizer.ggml.{name}_token_id")
                if token is not None and token not in special_ids:
                    raise ValueError("Model special token is not marked special by the tokenizer")
            if "tokenizer.ggml.merges" in metadata:
                ranks = getattr(tokenizer, "ranks", None)
                expected_merges = tuple(
                    tuple(x.split(" ")) if isinstance(x, str) else tuple(x)
                    for x in metadata["tokenizer.ggml.merges"]
                )
                actual_merges = (
                    tuple(pair for pair, _ in sorted(ranks.items(), key=lambda item: item[1]))
                    if ranks is not None
                    else None
                )
                if actual_merges != expected_merges:
                    raise ValueError("Tokenizer merge ranks do not match the model")
            self._tokenizer = tokenizer

    def token_inventory(self):
        """Decode ordered token bytes, with control tokens excluded from generated text."""
        with self._lock:
            self._check_open()
            if self._tokenizer is not None:
                tokenizer = self._tokenizer
                return tuple(
                    b"" if i in tokenizer.special_ids else tokenizer.token_bytes(i)
                    for i in range(self.vocab_size)
                )
            metadata = self.metadata
            words = metadata.get("tokenizer.ggml.tokens")
            if not isinstance(words, list) or len(words) != self.vocab_size:
                raise ValueError("Model token inventory is unavailable")
            kinds = metadata.get("tokenizer.ggml.token_type")
            if kinds is not None and (not isinstance(kinds, list) or len(kinds) != self.vocab_size):
                raise ValueError("Invalid model token types")
            special_ids = {i for i, kind in enumerate(kinds or ()) if kind in (2, 3, 4)}
            for key, value in metadata.items():
                if key.startswith("tokenizer.ggml.") and key.endswith("_token_id"):
                    if not isinstance(value, int) or not 0 <= value < self.vocab_size:
                        raise ValueError("Invalid model special token ID")
                    special_ids.add(value)
            if metadata.get("tokenizer.ggml.model") != "gpt2":
                raise ValueError("Attach a tokenizer to validate native token bytes")
            from .tokenizer import byte_mapping

            _, reverse = byte_mapping()
            try:
                return tuple(
                    b"" if i in special_ids else bytes(reverse[c] for c in word)
                    for i, word in enumerate(words)
                )
            except KeyError as error:
                raise ValueError("Model tokens require an explicit tokenizer") from error

    def encode(self, text, allow_special=False):
        if self._tokenizer is None:
            raise ValueError("Attach a validated tokenizer before text encoding")
        return self._tokenizer.encode(text, allow_special=allow_special)

    def decode_text(self, tokens, skip_special=True):
        if self._tokenizer is None:
            raise ValueError("Attach a validated tokenizer before text decoding")
        return self._tokenizer.decode(tokens, skip_special=skip_special)

    def _new_sequence(self):
        with self._lock:
            self._check_open()
            sequence = self._library.tvara_sequence_create(self._handle)
            if not sequence:
                self._raise_error()
            return sequence

    def _drop_sequence(self, sequence):
        with self._lock:
            if self._handle and self._library.tvara_sequence_drop(self._handle, sequence):
                self._raise_error()

    def logits(self, sequence, token):
        if (
            isinstance(token, (bool, np.bool_))
            or not isinstance(token, (int, np.integer))
            or not 0 <= token < self.vocab_size
        ):
            raise ValueError("Invalid token ID")
        output = np.empty(self.vocab_size, dtype=np.float32)
        with self._lock:
            self._check_open()
            status = self._library.tvara_sequence_decode(
                self._handle,
                sequence,
                int(token),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(output),
            )
            if status:
                self._raise_error()
        return output

    def iter_tokens(
        self,
        prompt,
        max_new_tokens=16,
        temperature=0,
        seed=0,
        eos_id=None,
        top_p=1,
        top_k=0,
        min_p=0,
        repetition=1,
        frequency=0,
        presence=0,
        token_constraint=None,
    ):
        prompt = list(prompt)
        if not prompt or any(
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, np.integer))
            or not 0 <= t < self.vocab_size
            for t in prompt
        ):
            raise ValueError("Invalid prompt token IDs")
        if (
            isinstance(max_new_tokens, (bool, np.bool_))
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
            or len(prompt) + max_new_tokens > self.context_size
        ):
            raise ValueError("Generation exceeds context limits")

        def real_scalar(value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError("Sampling parameters must be real scalars")
            try:
                return float(value)
            except (OverflowError, ValueError, TypeError) as error:
                raise ValueError("Sampling parameters must be finite real scalars") from error

        temperature, top_p, min_p, repetition, frequency, presence = (
            real_scalar(value)
            for value in (temperature, top_p, min_p, repetition, frequency, presence)
        )
        if (
            not np.isfinite(temperature)
            or temperature < 0
            or not np.isfinite(top_p)
            or not 0 < top_p <= 1
            or not np.isfinite(min_p)
            or not 0 <= min_p <= 1
        ):
            raise ValueError("Invalid sampling parameters")
        if (
            isinstance(top_k, (bool, np.bool_))
            or not isinstance(top_k, int)
            or top_k < 0
            or not np.isfinite(repetition)
            or repetition <= 0
            or not np.isfinite(frequency)
            or not np.isfinite(presence)
        ):
            raise ValueError("Invalid sampling filters or penalties")
        if eos_id is not None and (
            not isinstance(eos_id, (int, np.integer))
            or isinstance(eos_id, bool)
            or not 0 <= eos_id < self.vocab_size
        ):
            raise ValueError("Invalid EOS ID")
        if isinstance(seed, (bool, np.bool_)):
            raise ValueError("Invalid random seed")
        try:
            rng = np.random.default_rng(seed)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid random seed") from error
        sequence = self._new_sequence()
        history = prompt.copy()
        try:
            for token in prompt:
                logits = self.logits(sequence, token)
            for index in range(max_new_tokens):
                logits = logits.astype(np.float64, copy=True)
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite model logits")
                counts = np.bincount(history, minlength=self.vocab_size)
                seen = counts > 0
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    logits[seen] = np.where(
                        logits[seen] >= 0, logits[seen] / repetition, logits[seen] * repetition
                    )
                    logits -= counts * frequency + seen * presence
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite adjusted sampling logits")
                if token_constraint is not None:
                    logits = token_constraint.mask_logits(logits)
                if temperature == 0:
                    token = int(logits.argmax())
                else:
                    values = (logits.astype(float) - float(logits.max())) / temperature
                    p = np.exp(values)
                    p /= p.sum()
                    order = np.argsort(-p, kind="stable")
                    if top_k:
                        p[order[top_k:]] = 0
                    p[p < p.max() * min_p] = 0
                    p /= p.sum()
                    cumulative = np.cumsum(p[order])
                    p[order[np.r_[False, cumulative[:-1] >= top_p]]] = 0
                    p /= p.sum()
                    token = int(rng.choice(self.vocab_size, p=p))
                if token_constraint is not None:
                    token_constraint.accept(token)
                history.append(token)
                yield token
                if token == eos_id:
                    break
                if index + 1 < max_new_tokens:
                    logits = self.logits(sequence, token)
        finally:
            self._drop_sequence(sequence)

    def generate(self, prompt, **options):
        return list(self.iter_tokens(prompt, **options))

    # MODEL_METHODS
