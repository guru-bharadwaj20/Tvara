# Quantization reference and scalar kernels

Q8_0 stores a little-endian half scale plus 32 signed bytes (34 bytes total).
Q4_0 stores a half scale plus 16 packed bytes (18 total): low nibbles decode the
first 16 values and high nibbles the next 16, with integer offset -8.
Q4_K stores d and dmin half scales, 12 packed sub-block scale/minimum bytes, and
128 nibble bytes (144 total) for 256 weights. Eight 32-value groups use six-bit
scales and minima; dequantized values are d*scale*q - dmin*minimum.

Activation quantization uses one float32 scale per 32 values and signed bytes in
[-127,127]. Zero blocks retain zero scale. Activations must be finite and widths
divisible by 32. Rounding is nearest with halfway values away from zero. These
internal activation blocks are not serialized as GGUF Q8_0 weights.

The scalar dot implementations accumulate integer products and scale their sums.
Tests compare them against float32-dequantized weights and dequantized activation
blocks. This validates arithmetic for the chosen activation quantization; it does
not claim equivalence to unquantized activations. Checks cover known packed values,
signed extremes, subnormal halves, zero blocks, malformed sizes, and 100 seeded
random trials. Dot comparison tolerance is 1e-4*(1+abs(reference)); simple cases
also use tighter limits. Pretrained quantized-model quality has not been measured.
