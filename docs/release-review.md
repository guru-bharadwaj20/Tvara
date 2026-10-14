# Release review

`python tools/review_release.py` records this v0.1.0 review in
[bench/results/release-review.json](../bench/results/release-review.json); `--check`
repeats it without writing. It reads Git index blobs, not the working tree, and takes
licenses from installed metadata only. The run on 2026-10-14 (index staged on HEAD 53d9491, Python
3.13.15, Windows, with the `dev`, `server` and `oracle` extras installed) passed with
0 errors, 0 warnings and 5 info notes. Counts below come from that run.

## What ships

The release is the set of tracked files: 246 files, 2,401,255 bytes.

| Category | Files | License |
|---|---|---|
| Native source (`core/`, `bench/*.cpp`, CMake files) | 30 | MIT |
| Python package (`python/tvara`) | 22 | MIT |
| Tests (`tests/core`, `tests/python`) | 62 | MIT |
| Test fixtures (`tests/fixtures`) | 4 | MIT; upstream provenance below |
| Tools (`tools/`) | 31 | MIT |
| Benchmark results (`bench/results`) | 42 | MIT |
| Documentation (`*.md`) | 48 | MIT |
| CI workflows, build config, LICENSE | 6 | MIT |
| Third-party (`bench/third_party/stream.c`) | 1 | STREAM license |

The script fails on unclassified or unlisted third-party files, weight, dataset or key
files (`.gguf`, `.safetensors`, `.bin`, `.pt`, `.onnx`, `.npz`, `.pem`, `.env`,
`id_rsa*`, `tokenizer.json` and similar), files over 5 MB, unreviewed binaries, and
token or private-key patterns. The one allowlisted binary,
`tests/fixtures/reference-logits.npz` (1,188,076 bytes), is checked to hold only
`tokens_N`/`logits_N` arrays without pickled objects. A one-off scan of all 266 commits
found no weight, key, oversized blob or token pattern in history.

`.gitignore` excludes `/models/` (downloaded or generated weights and tokenizers),
`*.gguf`, `*.safetensors`, `/_work/` (llama.cpp checkout), `/build/`, `/.venv/`, `.env`,
bytecode, egg-info and tool caches.

## Python dependencies

None are vendored; pip installs them from PyPI. Markers were evaluated for Python 3.13
on Windows, so other platforms may resolve slightly different transitive sets.

| Group | Direct (license) | Transitive (license) |
|---|---|---|
| runtime | numpy 2.5.4 (BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0), regex 2026.9.29 (Apache-2.0 AND CNRI-Python), jsonschema 4.26.0 (MIT) | attrs, jsonschema-specifications, referencing, rpds-py (MIT) |
| dev | cmake 3.31.6 (Apache 2.0; BSD classifier), pytest 8.4.2 (MIT), ruff 0.11.13 (MIT) | iniconfig, pluggy (MIT); Pygments (BSD-2-Clause); packaging (Apache-2.0 OR BSD-2-Clause); colorama (BSD) |
| oracle | torch 2.14.1 (Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT), transformers 4.57.6 (Apache 2.0), safetensors 0.8.0 (Apache), tokenizers 0.22.2 (Apache) | huggingface_hub, requests (Apache); filelock, PyYAML, urllib3, charset-normalizer, setuptools (MIT); fsspec, networkx, sympy, mpmath, idna, MarkupSafe (BSD); typing_extensions (PSF-2.0); certifi (MPL-2.0); tqdm (MPL-2.0 AND MIT) |
| server | fastapi 0.143.0 (MIT), httpx 0.28.1 (BSD-3-Clause), Jinja2 3.1.6 (BSD), uvicorn 0.54.0 (BSD-3-Clause) | annotated-doc, annotated-types, anyio, h11, pydantic, pydantic_core, typing-inspection (MIT); click, httpcore, idna, MarkupSafe, starlette (BSD-3-Clause); opentelemetry-api (Apache-2.0); typing_extensions (PSF-2.0); certifi (MPL-2.0) |

The only copyleft licenses are MPL-2.0 (file-level): certifi, reached through httpx in
the optional `server` extra and through requests in `oracle`, and tqdm (MPL-2.0 AND MIT)
in `oracle`. Tvara neither modifies nor redistributes either. `safetensors` and
`tokenizers`, previously imported without being declared, are now listed in the `oracle`
extra. `anyio`, `pydantic` and `starlette` are imported directly but declared only
through fastapi and httpx. CI uses `actions/checkout@v4`, `actions/setup-python@v5`,
`actions/upload-artifact@v4` and `msys2/setup-msys2@v2`; these run on GitHub and are
not redistributed.

## STREAM obligations

`bench/third_party/stream.c` is STREAM 5.10, copyright John D. McCalpin, 1991-2013,
under its embedded license, not MIT. Its LF-normalized content matches upstream (SHA-256
`a52bae5e...`, compared 2026-10-14); the script pins that hash and the license header.
The terms allow use, redistribution and modification, including commercial use. Only
results that follow the STREAM Run Rules, as judged by the copyright holder, may be
called "STREAM benchmark results"; results from modified source or non-conforming runs
must be clearly labelled when published, e.g. "based on a variant of the STREAM
benchmark code". Use constitutes acceptance; there is no warranty.

`bench/results/memory-bandwidth.json` is a local single-thread run with only
`STREAM_ARRAY_SIZE` and `NTIMES` set at compile time. Run Rules conformance has not been
reviewed, so publish it labelled as not a conformance-reviewed STREAM result. Its
`source_sha256` (`4802cae6...`) hashes the CRLF Windows checkout, not the LF blob.

## Upstream models and external code

| Item | Use | Tracked content | License |
|---|---|---|---|
| hf-internal-testing/tiny-random-LlamaForCausalLM @ 9fb1912 | Random-weight logit oracle | Manifest and derived logits only | None declared; card reads "[More Information Needed]" |
| Qwen/Qwen2.5-0.5B-Instruct `tokenizer.json` | Tokenizer ID tests | Repository name and SHA-256 only | Apache-2.0 on model card |
| Benchmark and API GGUF fixtures | Matched baseline, server tests | Benchmark checksums only | Generated locally from seed 2026; original |
| llama.cpp @ 45451c3 | Matched CPU baseline | Revision only | MIT at that revision |
| Llama 3.2, Qwen 2.5, SmolLM2, Gemma 3 weights | Targets in [models.md](models.md) | None | Not used; upstream terms apply when used |

Upstream entries were checked against Hugging Face and GitHub on 2026-10-14; Qwen
revision 7ae5576 serves the pinned tokenizer SHA-256. No weights or tokenizers are tracked.

## Residual risks

- The tiny-random oracle declares no license; only its random-weight logits ship.
  Replacing it with a locally generated checkpoint would remove the dependency.
- torch's license metadata summarizes bundled third-party code; its upstream NOTICE
  files govern redistribution of torch itself, which Tvara does not do.
- The Qwen tokenizer manifest records a checksum but not a revision.
- Licenses come from package metadata; secret scanning is pattern-based and the
  history scan was a one-off.
