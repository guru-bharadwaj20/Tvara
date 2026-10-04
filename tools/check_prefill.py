"""Compile and run a native optimization check, preserving measured JSON evidence."""

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("result")
    args = parser.parse_args()
    compiler = os.environ.get("CXX", shutil.which("g++") or "g++")
    with tempfile.TemporaryDirectory(prefix="tvara-prefill-") as directory:
        binary = str(pathlib.Path(directory) / "check.exe")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Icore/include",
                "-Itests/core",
                args.source,
                "-o",
                binary,
            ],
            check=True,
        )
        run = subprocess.run([binary], check=True, capture_output=True, text=True)
    measured = json.loads(run.stdout)
    measured.update(
        platform=platform.platform(),
        compiler=subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        flags="-std=c++20 -O2",
    )
    result = pathlib.Path(args.result)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(measured))


if __name__ == "__main__":
    main()
