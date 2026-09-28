# GGUF loader

The native reader accepts little-endian GGUF v2/v3 with bounded scalar metadata,
strings, and nonnested arrays. Tensor ranks are 1–4. It checks duplicate keys/names,
count and length limits, dimension overflow, alignment, storage overlap, and payload
extent. General alignment defaults to 32 and must be a power of two up to 65536.

Recognized storage layouts are F32, F16, Q4_0, Q8_0, and Q4_K. Quantized first
dimensions must be divisible by their block size. Recognizing a storage layout
does not yet mean model execution can consume it. Unsupported tensor formats fail
at load time rather than receiving guessed byte lengths.

MappedFile is read-only and owns OS handles through RAII. TensorView retains a
shared mapping owner, so a view remains valid after its ModelFile wrapper is freed.
Metadata parsing produces owned strings and arrays. Views never allow weight writes.

Llama configuration mapping validates required dimensions and tensor names. Scaled
RoPE and other architectures are explicitly unsupported in this initial mapping.
Truncation, malformed metadata, unsupported versions/types, and missing model
tensors raise tvara::gguf::Error. No claim of arbitrary GGUF compatibility is made.

Reproduce the native malformed-file and mapping tests with ctest --preset debug.
The format reference is the upstream GGUF specification linked in references.md.
