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
        if len(tokenizer.vocabulary) != self.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match model")
        self._tokenizer = tokenizer

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
        if not isinstance(token, (int, np.integer)) or not 0 <= token < self.vocab_size:
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
    ):
        prompt = list(prompt)
        if not prompt or any(
            not isinstance(t, (int, np.integer)) or not 0 <= t < self.vocab_size for t in prompt
        ):
            raise ValueError("Invalid prompt token IDs")
        if (
            not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
            or len(prompt) + max_new_tokens > self.context_size
        ):
            raise ValueError("Generation exceeds context limits")
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
            not isinstance(top_k, int)
            or top_k < 0
            or not np.isfinite(repetition)
            or repetition <= 0
            or not np.isfinite(frequency)
            or not np.isfinite(presence)
        ):
            raise ValueError("Invalid sampling filters or penalties")
        if eos_id is not None and not 0 <= eos_id < self.vocab_size:
            raise ValueError("Invalid EOS ID")
        rng = np.random.default_rng(seed)
        sequence = self._new_sequence()
        history = prompt.copy()
        try:
            for token in prompt:
                logits = self.logits(sequence, token)
            for index in range(max_new_tokens):
                logits = logits.copy()
                counts = np.bincount(history, minlength=self.vocab_size)
                seen = counts > 0
                logits[seen] = np.where(
                    logits[seen] >= 0, logits[seen] / repetition, logits[seen] * repetition
                )
                logits -= counts * frequency + seen * presence
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
