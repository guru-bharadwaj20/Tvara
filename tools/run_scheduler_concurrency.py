"""Measure native scheduler latency at several concurrency levels (T263).

Concurrency N means N requests are submitted together and the scheduler may hold
all N active at once (max_active=N). Each request has a fixed-length seeded prompt
and a fixed output length so levels are comparable. Runs the real native engine
through NativeSchedulerBackend; decode is serial inside each batch.
"""

import argparse
import datetime
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from tvara.native import NativeModel
from tvara.scheduler import NativeSchedulerBackend, Request, Scheduler

LEVELS = (1, 2, 4, 8, 16)


def make_requests(count, seed, prompt_tokens, max_tokens, vocab=257):
    rng = random.Random(seed)
    return [
        Request(
            f"req-{i}",
            tuple(rng.randrange(vocab) for _ in range(prompt_tokens)),
            max_tokens,
        )
        for i in range(count)
    ]


def pct(values):
    if not values:
        return {"count": 0, "p50_seconds": None, "p99_seconds": None}
    return {
        "count": len(values),
        "p50_seconds": float(np.percentile(values, 50)),
        "p99_seconds": float(np.percentile(values, 99)),
    }


def run_level(model, level, seed, prompt_tokens, max_tokens, prefill_chunk):
    requests = make_requests(level, seed, prompt_tokens, max_tokens)
    scheduler = Scheduler(
        NativeSchedulerBackend(model), max_active=level, prefill_chunk=prefill_chunk
    )
    start = time.perf_counter()
    for r in requests:
        r.submitted = start
        scheduler.submit(r)
    scheduler.run()
    elapsed = time.perf_counter() - start
    ttft = [r.token_times[0] - r.submitted for r in requests if r.token_times]
    itl = [b - a for r in requests for a, b in zip(r.token_times, r.token_times[1:])]
    tokens = sum(len(r.output) for r in requests)
    return {
        "seconds": elapsed,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed,
        "ttft_samples": ttft,
        "inter_token_samples": itl,
        "outputs": [r.output for r in requests],
    }


def measure(
    model_path="models/api-fixture/model.gguf",
    levels=LEVELS,
    repetitions=5,
    seed=263,
    prompt_tokens=16,
    max_tokens=16,
    prefill_chunk=8,
    out="bench/results/scheduler-concurrency.json",
):
    measured_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = []
    with NativeModel(model_path) as model:
        # Warm-up run, discarded.
        run_level(model, 1, seed, prompt_tokens, max_tokens, prefill_chunk)
        for level in levels:
            runs = [
                run_level(model, level, seed, prompt_tokens, max_tokens, prefill_chunk)
                for _ in range(repetitions)
            ]
            if any(r["outputs"] != runs[0]["outputs"] for r in runs):
                raise RuntimeError(f"Non-deterministic outputs at concurrency {level}")
            ttft = [x for r in runs for x in r["ttft_samples"]]
            itl = [x for r in runs for x in r["inter_token_samples"]]
            tps = [r["tokens_per_second"] for r in runs]
            rows.append(
                {
                    "concurrency": level,
                    "ttft": pct(ttft),
                    "inter_token": pct(itl),
                    "throughput_tokens_per_second": {
                        "median": float(np.median(tps)),
                        "min": float(min(tps)),
                        "max": float(max(tps)),
                    },
                    "raw": [{k: v for k, v in r.items() if k != "outputs"} for r in runs],
                }
            )
    result = {
        "task": "T263",
        "measured_at": measured_at,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": model_path,
        "fixture": "random native GGUF fixture, not pretrained",
        "backend": "NativeSchedulerBackend: serial per-sequence decode within each batch, greedy",
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "prefill_chunk": prefill_chunk,
        "max_active": "equal to concurrency",
        "repetitions": repetitions,
        "warmup_runs_discarded": 1,
        "percentiles": "numpy linear interpolation over pooled samples from all repetitions",
        "rows": rows,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/api-fixture/model.gguf")
    parser.add_argument("--levels", default=",".join(map(str, LEVELS)))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument("--out", default="bench/results/scheduler-concurrency.json")
    args = parser.parse_args()
    result = measure(
        args.model,
        tuple(int(x) for x in args.levels.split(",")),
        args.repetitions,
        args.seed,
        out=args.out,
    )
    for row in result["rows"]:
        print(
            f"c={row['concurrency']:>2} "
            f"ttft p50={row['ttft']['p50_seconds'] * 1e3:.2f}ms "
            f"p99={row['ttft']['p99_seconds'] * 1e3:.2f}ms "
            f"itl p50={row['inter_token']['p50_seconds'] * 1e3:.3f}ms "
            f"p99={row['inter_token']['p99_seconds'] * 1e3:.3f}ms "
            f"tps={row['throughput_tokens_per_second']['median']:.1f}"
        )


if __name__ == "__main__":
    main()
