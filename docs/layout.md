# Intended repository layout

| Path | Responsibility |
|---|---|
| core/include/tvara | Public C++ and stable C ABI declarations |
| core/src | Portable kernels, model execution, GGUF and cache implementations |
| python/tvara | Python bindings and float32 reference/oracle tooling |
| server | HTTP integration and request/stream handling |
| tests/core | Native correctness and storage checks |
| tests/python | Numerical oracle, tokenizer, and integration tests |
| bench | Reproducible benchmark and load-generation commands |
| docs | Decisions, design contracts, proofs, and measured results |
| .github/workflows | Portable build and correctness CI |

Build outputs belong in build/, virtual environments in .venv/, and downloaded
weights in models/. Generated data and developer execution helpers stay untracked.
Create source files when a component has an actual implementation; avoid empty
directory placeholders suggesting that a subsystem already exists.
