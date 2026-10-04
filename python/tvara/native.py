"""Owned ctypes handles for Tvara's CPU-only native model library."""

import ctypes
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
    names = ["tvara_capi.dll", "libtvara_capi.so", "libtvara_capi.dylib"]
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

    # MODEL_METHODS
