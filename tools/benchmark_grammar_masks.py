"""Local mask benchmark; fixture-driven logits do not measure model semantic accuracy."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from jsonschema import validate
from tvara.tokenizer import ByteBPE, byte_mapping
from tvara.response_constraints import response_constraint


def measure():
    forward, _ = byte_mapping()
    tokenizer = ByteBPE(list(forward.values()), [])
    schema = {
        "type": "object",
        "properties": {"city": {"enum": ["Paris", "東京"]}, "ok": {"type": "boolean"}},
        "required": ["city", "ok"],
        "additionalProperties": False,
    }
    format = {"type": "json_schema", "json_schema": {"schema": schema}}
    latencies = []
    outputs = []
    hits = 0
    misses = 0
    for city in ("Paris", "東京"):
        for flag in (True, False):
            expected = json.dumps(
                {"city": city, "ok": flag}, ensure_ascii=True, separators=(",", ":")
            )
            constraint = response_constraint(format, tokenizer)
            emitted = []
            for wanted in tokenizer.encode(expected):
                logits = np.linspace(-1, 1, len(tokenizer.vocabulary))
                logits[wanted] = 10
                begin = time.perf_counter_ns()
                masked = constraint.mask_logits(logits)
                latencies.append(time.perf_counter_ns() - begin)
                token = int(masked.argmax())
                constraint.accept(token)
                emitted.append(token)
            assert constraint.complete
            output = tokenizer.decode(emitted, errors="strict")
            validate(json.loads(output), schema)
            outputs.append(output)
            constraint.engine.mask(constraint.state)
            constraint.engine.mask(constraint.state)
            hits += constraint.engine.hits
            misses += constraint.engine.misses
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "vocabulary_size": 256,
        "schema_outputs": outputs,
        "valid_outputs": len(outputs),
        "latency_ns": {
            "median": statistics.median(latencies),
            "p95": float(np.percentile(latencies, 95)),
            "samples": len(latencies),
        },
        "cache_hits": hits,
        "cache_misses": misses,
        "scope": "Fixture-driven constrained greedy sampling; not trained-model quality or production vocabulary throughput",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench/results/grammar-masks.json")
    args = parser.parse_args()
    result = measure()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps({"valid_outputs": result["valid_outputs"], "latency_ns": result["latency_ns"]})
    )


if __name__ == "__main__":
    main()
