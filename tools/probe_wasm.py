"""Record local WebAssembly toolchain availability without claiming a wasm build."""

import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    compiler = shutil.which("em++")
    data = dict(
        host=platform.platform(),
        emscripten_available=bool(compiler),
        status="toolchain_available_unvalidated" if compiler else "toolchain_unavailable",
        blockers=[
            "native file mapping requires a wasm/virtual-file-system path",
            "thread pool requires pthread build plus browser cross-origin isolation",
            "runtime x86 intrinsics must remain excluded from wasm",
            "browser memory budgets and model download paths need explicit limits",
        ],
        recommended_scope="standalone scalar and wasm-SIMD Q8 kernels before complete runtime",
    )
    if compiler:
        data["compiler"] = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    path = Path("bench/results/variants/wasm-feasibility.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
