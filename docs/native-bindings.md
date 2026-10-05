# Native Python bindings

Build the shared `tvara_capi` target before constructing `NativeModel`. Set
`TVARA_LIBRARY` or pass an explicit library path when using a separate checkout.
MinGW produces `libtvara_capi.dll`; the library finder also recognizes the other
platform names. MinGW runtime libraries are linked statically for ctypes loading.

The C ABI exposes opaque model handles and sequence IDs, caller-owned float32
logit buffers, and a thread-local copied error message. Python owns the handle,
serializes calls, closes it through a context manager or finalizer, and validates
token IDs and buffer sizes. Closing a generator releases its sequence's KV pages.

Attach a validated tokenizer with exactly matching ordered GGUF vocabulary, declared control-token IDs and known BPE merge ranks before
encoding text. Generation uses native paged decode and Python sampling. An explicit
seed reproduces results within the Python sampler; it does not promise identical
draws to the separate C++ standard-library sampler.

Local validation executes three real-library checks: model loading/closing,
metadata/tokenizer attachment, and generation matching the six-token Hugging Face
greedy fixture, including seeded sampling and generator cleanup. These tests skip
only when their optional checkpoint or built library is unavailable.

Derive speculative identity with `tokenizer_identity(tokenizer, eos_id)` after attachment; it includes tokenizer normalization, merge and special-token policy. Unattached manually declared identities require caller-provided configuration information where GGUF lacks encoder details. The Python wheel contains Python modules only: build a matching native library from the source checkout/archive and set `TVARA_LIBRARY` or an explicit library path.
