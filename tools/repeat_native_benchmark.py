"""Repeated native timing; do not infer thermal conditions from elapsed time."""

from datetime import datetime, timezone
import importlib.util
import json
import statistics
from pathlib import Path


def main():
    spec = importlib.util.spec_from_file_location(
        "matched", Path(__file__).with_name("run_matched_baseline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = [module.native_run("models/benchmark-f32/model.gguf") for _ in range(3)]
    for run in runs:
        run.pop("prefill_logits")
    summary = {}
    for name in ("prefill_seconds", "decode_seconds"):
        values = [value for run in runs for value in run[name]]
        summary[name] = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "samples": len(values),
        }
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "summary": summary,
        "conditions": {
            "temperature_celsius": None,
            "frequency_hz": None,
            "cooldown": "not imposed",
            "concurrent_load": "not controlled; build processes and other applications may be active",
        },
        "scope": "Three native runs with one warmup and five measured samples each; tiny random-weight model, ctypes calls included. Temperature telemetry unavailable, so thermal stability is not established.",
    }
    path = Path("bench/results/repeated-native.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
