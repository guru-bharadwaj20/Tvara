"""Record local timings with actual execution time, separate from commit dates."""

import datetime
import json
from pathlib import Path
import platform
import subprocess

binary = Path("build/release/kernel_overhead.exe")
if not binary.exists():
    binary = Path("build/release/kernel_overhead")
result = json.loads(subprocess.check_output([str(binary)], text=True))
result.update(
    executed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    platform=platform.platform(),
    processor=platform.processor(),
    build="Release",
    limitation="single local microbenchmark run; not model throughput or a comparative speed claim",
)
target = Path("bench/results/kernel-overhead.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))
