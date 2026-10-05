"""Run the upstream STREAM 5.10 source without modifying its loops."""

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    source = Path("bench/third_party/stream.c")
    binary = Path("build/bench-stream" + (".exe" if os.name == "nt" else ""))
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "gcc",
        "-O3",
        "-DSTREAM_ARRAY_SIZE=8388608",
        "-DNTIMES=10",
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    output = subprocess.check_output([str(binary.resolve())], text=True)
    rows = {}
    for line in output.splitlines():
        match = re.match(
            r"^(Copy|Scale|Add|Triad):\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line
        )
        if match:
            rows[match[1].lower()] = {
                "best_mb_per_second": float(match[2]),
                "average_seconds": float(match[3]),
                "minimum_seconds": float(match[4]),
                "maximum_seconds": float(match[5]),
            }
    if len(rows) != 4 or "Solution Validates" not in output:
        raise RuntimeError("STREAM validation failed")
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.cs.virginia.edu/stream/FTP/Code/stream.c",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compile_command": command,
        "array_elements": 8388608,
        "working_set_bytes": 201326592,
        "threads": 1,
        "results": rows,
        "raw_output": output,
        "scope": "Unmodified STREAM 5.10 source, local single-thread run; uncontrolled frequency and temperature; no certified-conformance or model-throughput claim",
    }
    target = Path("bench/results/memory-bandwidth.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
