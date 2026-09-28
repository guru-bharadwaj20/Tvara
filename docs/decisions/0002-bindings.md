# Decision 0002: Python bindings

Use Python's standard-library ctypes with a versioned C ABI. Pass pointers and
explicit element counts; copy error messages before releasing an owned handle.
Do not expose C++ containers, references, exceptions, or ownership to Python.

The local compiler is MinGW while installed CPython uses a different extension
toolchain. A C ABI avoids depending on pybind11/PyO3 extension builds during the
portable milestone. ctypes is less ergonomic: array dtype, contiguity, pointer
lifetime, and error conversion need explicit wrappers and tests.

Bindings will first cover real native kernels, then model handles and generation.
Choosing a binding mechanism does not mean those later interfaces exist yet.
