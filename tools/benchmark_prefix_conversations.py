"""Actual native cached/cold inference, with exact final-logit checks."""

from datetime import datetime, timezone
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    binary = Path("build/prefix-conversations" + (".exe" if os.name == "nt" else ""))
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-Icore/include",
            "bench/prefix_conversations.cpp",
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = json.loads(
        subprocess.check_output(
            [str(binary.resolve()), "models/benchmark-f32/model.gguf"], text=True
        )
    )
    result.update(
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Twelve fixed token conversation continuations over one random tiny model prefix; cached/cold final logits identical. Cold forward includes sequence lifecycle; warm time includes lookup/restore/decode and excludes sequence drop. Initial cache fill excluded; not trained conversations or multi-tenant throughput.",
    )
    measured = result["traces"][1:]
    result["warmup_traces"] = 1
    result["median_cold_seconds"] = statistics.median(r["cold_seconds"] for r in measured)
    result["median_cached_seconds"] = statistics.median(r["cached_seconds"] for r in measured)
    path = Path("bench/results/prefix-conversations.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: result[k] for k in ("median_cold_seconds", "median_cached_seconds")}))


if __name__ == "__main__":
    main()
