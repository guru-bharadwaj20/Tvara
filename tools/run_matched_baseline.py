"""Matched CPU-only llama.cpp/native timing and independent final-logit comparison."""

from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from tvara.native import NativeModel

REVISION = "45451c366d4256f94fbc8845645e31a6b9155054"


def build_baseline():
    source = Path(os.environ.get("TVARA_LLAMA_SOURCE", "_work/llama.cpp")).resolve()
    if not source.exists():
        raise FileNotFoundError(
            "Clone pinned llama.cpp into _work/llama.cpp before running this baseline"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Unexpected llama.cpp source revision")
    patch = '\nif(DEFINED TVARA_MATCHED_BENCH_SOURCE)\n add_executable(tvara-matched-bench "${TVARA_MATCHED_BENCH_SOURCE}")\n target_link_libraries(tvara-matched-bench PRIVATE llama)\nendif()\n'
    cmake = source / "CMakeLists.txt"
    text = cmake.read_text(encoding="utf8")
    if "TVARA_MATCHED_BENCH_SOURCE" not in text:
        cmake.write_text(text + patch, encoding="utf8")
    build = Path(os.environ.get("TVARA_LLAMA_BUILD", "_work/llama-build")).resolve()
    cmake_binary = Path(sys.executable).with_name("cmake.exe" if os.name == "nt" else "cmake")
    cmake_command = str(cmake_binary) if cmake_binary.exists() else "cmake"
    command = [
        cmake_command,
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=ON",
        "-DGGML_OPENMP=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DGGML_CCACHE=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DTVARA_MATCHED_BENCH_SOURCE=" + Path("bench/llama_matched.cpp").resolve().as_posix(),
    ]
    if os.name == "nt" and not (build / "CMakeCache.txt").exists():
        command += ["-G", "MinGW Makefiles"]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            cmake_command,
            "--build",
            str(build),
            "--target",
            "tvara-matched-bench",
            "--parallel",
            "2",
        ],
        check=True,
    )
    binary = build / (
        "bin/tvara-matched-bench.exe" if os.name == "nt" else "bin/tvara-matched-bench"
    )
    return binary, revision


def release_library():
    for name in ("libtvara_capi.dll", "tvara_capi.dll", "libtvara_capi.so", "libtvara_capi.dylib"):
        path = Path("build/release") / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Build the native Release library before timing; Debug fallback is not allowed"
    )


def native_run(path):
    library = release_library()
    with NativeModel(path, library=library) as model:
        prefill = []
        decode = []
        reference = None
        for repeat in range(6):
            sequence = model._new_sequence()
            try:
                begin = time.perf_counter()
                for token in range(1, 17):
                    logits = model.logits(sequence, token)
                pp = time.perf_counter() - begin
                if repeat == 0:
                    reference = logits.tolist()
                begin = time.perf_counter()
                for token in range(17, 25):
                    model.logits(sequence, token)
                tg = time.perf_counter() - begin
                if repeat:
                    prefill.append(pp)
                    decode.append(tg)
            finally:
                model._drop_sequence(sequence)
    return {
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "prefill_logits": reference,
        "native_library": {
            "name": library.name,
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "build": "Release",
        },
    }


def main():
    binary, revision = build_baseline()
    rows = []
    for kind in ("f32", "q8"):
        model = Path("models/benchmark-" + kind + "/model.gguf")
        if not model.exists():
            raise FileNotFoundError("Run tools/pin_benchmark_models.py first")
        baseline = json.loads(
            subprocess.check_output([str(binary), str(model.resolve())], text=True)
        )
        native = native_run(model)
        error = float(
            np.max(np.abs(np.asarray(native["prefill_logits"]) - baseline["prefill_logits"]))
        )
        # Quantized llama.cpp can also quantize activations; its arithmetic differs from native dequantized F32 execution.
        if kind == "f32" and not np.allclose(
            native["prefill_logits"], baseline["prefill_logits"], atol=1e-5, rtol=1e-4
        ):
            raise AssertionError("Independent F32 logits mismatch")
        for result in (baseline, native):
            result.pop("prefill_logits")
        rows.append(
            {
                "model": str(model).replace("\\", "/"),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "quantization": kind,
                "llama_cpp": baseline,
                "tvara_native": native,
                "maximum_logit_error": error,
                "f32_oracle_pass": kind == "f32",
            }
        )
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "llama_cpp_revision": revision,
        "threads": 1,
        "kv_precision": "F32 keys and values in both engines",
        "output_policy": "Full vocabulary logits at every prefill and decode position in both engines",
        "prompt_tokens": list(range(1, 17)),
        "decode_tokens": list(range(17, 25)),
        "warmup_runs": 1,
        "measured_runs": 5,
        "rows": rows,
        "scope": "Same random-weight interleaved-RoPE model files and exact token IDs; tiny-model ctypes/allocation overhead matters; Q8 native weights dequantize to F32 and llama.cpp uses quantized kernels; no pretrained quality or large-model speed claim",
    }
    target = Path("bench/results/matched-baseline.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps({"rows": len(rows), "f32_maximum_logit_error": rows[0]["maximum_logit_error"]})
    )


if __name__ == "__main__":
    main()
