# Native Python bindings

Build the shared `tvara_capi` target before constructing `NativeModel`. Set
`TVARA_LIBRARY` or pass an explicit library path when using a separate checkout.
MinGW produces `libtvara_capi.dll`; the library finder also recognizes the other
platform names. MinGW runtime libraries are linked statically for ctypes loading.

The C ABI exposes opaque model handles and sequence IDs, caller-owned float32
logit buffers, and a thread-local copied error message. Python owns the handle,
serializes calls, closes it through a context manager or finalizer, and validates
token IDs and buffer sizes. Closing a generator releases its sequence's KV pages.

Attach an independently validated tokenizer with the same vocabulary size before
encoding text. Generation uses native paged decode and Python sampling. An explicit
seed reproduces results within the Python sampler; it does not promise identical
draws to the separate C++ standard-library sampler.

Local validation executes three real-library checks: model loading/closing,
metadata/tokenizer attachment, and generation matching the six-token Hugging Face
greedy fixture, including seeded sampling and generator cleanup. These tests skip
only when their optional checkpoint or built library is unavailable.
