"""Measure native speculative decoding speedups with paired repeated runs (T264).

The target is an eight-layer, 512-wide random-weight Llama fixture. The draft is the same
seeded fixture truncated to its first layer, so it shares the embeddings, output projection
and first block (early-exit self-drafting). Two regimes vary only the output projection
scale: "peaked" (std 0.5, low-entropy and weakly agreeing distributions) and "flat" (std
0.05, near-uniform and strongly agreeing distributions), bracketing acceptance. Prompt lookup proposes n-gram continuations from
the history and costs no draft model evaluation. All arms sample from the target at
temperature 1 through the same Python/ctypes per-token path, so the baseline differs only
in the decoding algorithm. Each seed runs the baseline and the speculative arm back to
back in alternating order; statistics are over seeds. Weights are random, so neither
acceptance nor speed represents trained-model text, code or JSON behavior.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_api_fixture import make_fixture  # noqa: E402
from tvara.native import NativeModel  # noqa: E402
from tvara.speculative import (  # noqa: E402
    DraftController,
    NativeSpeculativeModel,
    TokenizerIdentity,
    propose_lookup,
    speculative_generate,
)

IDENTITY = TokenizerIdentity(tuple(bytes([i]) for i in range(256)) + (b"",), "byte-fixture-v1")
FIXTURE = dict(hidden=512, ffn=1536, heads=8, kv_heads=4, context=256)
REGIMES = {"peaked": 0.5, "flat": 0.05}
PROMPTS = {
    "text": "The small model reads the prompt and",
    "code": "def add(a, b):\n    return a +",
    "json": '{"name": "Ada", "id": 7, "tags": [',
    "repetitive": "abc abc abc abc abc abc abc abc",
}


class HistoryDraft:
    """Prompt-lookup draft state: token history only, no model evaluation."""

    identity = IDENTITY

    def __init__(self, prompt):
        self.history = list(prompt)

    def advance(self, token):
        self.history.append(token)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, checkpoint):
        del self.history[checkpoint:]


def baseline(target_model, prompt, tokens, seed):
    target = NativeSpeculativeModel(target_model, prompt, IDENTITY)
    try:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        for _ in range(tokens):
            p = target.distribution()
            target.advance(int(rng.choice(len(p), p=p)))
        return time.perf_counter() - start, tokens
    finally:
        target.close()


def speculative(target_model, draft_model, prompt, tokens, seed, arm):
    target = NativeSpeculativeModel(target_model, prompt, IDENTITY)
    draft = (
        HistoryDraft(prompt)
        if arm["proposer"] == "lookup"
        else NativeSpeculativeModel(draft_model, prompt, IDENTITY)
    )
    k = arm["length"]
    controller = (
        DraftController() if k == "adaptive" else DraftController(length=k, minimum=k, maximum=k)
    )
    steps = []
    observe = controller.observe

    def record(result):
        steps.append((result.proposed, result.accepted))
        return observe(result)

    controller.observe = record
    try:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        output = speculative_generate(
            draft,
            target,
            tokens,
            rng,
            controller,
            proposer=propose_lookup if arm["proposer"] == "lookup" else None,
        )
        elapsed = time.perf_counter() - start
    finally:
        target.close()
        if isinstance(draft, NativeSpeculativeModel):
            draft.close()
    return {
        "seconds": elapsed,
        "emitted": len(output),
        "steps": len(steps),
        "proposed": sum(p for p, _ in steps),
        "accepted": sum(a for _, a in steps),
    }


def bootstrap_median(values, rng, resamples=10000):
    values = np.asarray(values)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True)
    low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    return float(low), float(high)


def sign_test(differences):
    """Exact two-sided binomial sign test p-value, ties dropped."""
    from math import comb

    wins = int(np.sum(np.asarray(differences) > 0))
    n = int(np.sum(np.asarray(differences) != 0))
    if not n:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, n - wins) + 1)) / 2**n
    return min(1.0, 2 * tail)


def entropy_bits(model, prompt):
    sequence = NativeSpeculativeModel(model, prompt, IDENTITY)
    try:
        p = sequence.distribution()
    finally:
        sequence.close()
    nonzero = p[p > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def summarize(arm, prompt_name, runs, rng):
    ratios = [r["baseline_per_token"] / r["speculative_per_token"] for r in runs]
    proposed = sum(r["proposed"] for r in runs)
    accepted = sum(r["accepted"] for r in runs)
    per_seed_acceptance = [r["accepted"] / r["proposed"] for r in runs if r["proposed"]]
    acceptance_ci = (
        bootstrap_median(per_seed_acceptance, rng) if per_seed_acceptance else (None, None)
    )
    low, high = bootstrap_median(ratios, rng)
    return {
        "arm": arm["name"],
        "prompt": prompt_name,
        "seeds": len(runs),
        "proposed": proposed,
        "accepted": accepted,
        "acceptance_rate_pooled": accepted / proposed if proposed else None,
        "acceptance_rate_median_seed": (
            float(np.median(per_seed_acceptance)) if per_seed_acceptance else None
        ),
        "acceptance_rate_median_ci95": list(acceptance_ci),
        "tokens_per_target_step_mean": float(np.mean([r["emitted"] / r["steps"] for r in runs])),
        "speedup_median": float(np.median(ratios)),
        "speedup_ci95": [low, high],
        "speedup_min": float(min(ratios)),
        "speedup_max": float(max(ratios)),
        "seeds_faster": int(sum(x > 1 for x in ratios)),
        "sign_test_p": sign_test([x - 1 for x in ratios]),
        "baseline_ms_per_token_median": float(
            np.median([r["baseline_per_token"] for r in runs]) * 1e3
        ),
        "speculative_ms_per_token_median": float(
            np.median([r["speculative_per_token"] for r in runs]) * 1e3
        ),
    }


ARMS = [{"name": f"draft k={k}", "proposer": "draft", "length": k} for k in (1, 2, 4)] + [
    {"name": "draft adaptive", "proposer": "draft", "length": "adaptive"},
    {"name": "lookup k=4", "proposer": "lookup", "length": 4},
]


def run_regime(regime, output_std, args, stats_seed):
    stats_rng = np.random.default_rng(stats_seed)
    prompts = {name: list(text.encode()) for name, text in PROMPTS.items()}
    layer_counts = {"target": args.target_layers, "draft": args.draft_layers}
    paths = {
        role: make_fixture(
            Path(args.fixtures) / f"{regime}-{role}",
            layers=layers,
            output_std=output_std,
            **FIXTURE,
        )
        / "model.gguf"
        for role, layers in layer_counts.items()
    }
    fixtures = {
        role: {
            "path": str(path).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "layers": layer_counts[role],
            **FIXTURE,
            "output_std": output_std,
            "vocabulary": 257,
            "seed": 2026,
        }
        for role, path in paths.items()
    }
    results, raw = [], []
    with NativeModel(paths["target"]) as target_model, NativeModel(paths["draft"]) as draft_model:
        library = str(Path(target_model._library._name).resolve().relative_to(Path.cwd()))
        entropy = {
            name: {
                "target": entropy_bits(target_model, prompt),
                "draft": entropy_bits(draft_model, prompt),
            }
            for name, prompt in prompts.items()
        }
        # Warm-up, discarded.
        baseline(target_model, prompts["text"], args.tokens, 0)
        speculative(target_model, draft_model, prompts["text"], args.tokens, 0, ARMS[2])
        for arm in ARMS:
            for prompt_name, prompt in prompts.items():
                runs = []
                for seed in range(args.seeds):
                    order = ("baseline", "speculative") if seed % 2 else ("speculative", "baseline")
                    row = {"seed": seed, "order": order[0] + "-first"}
                    for which in order:
                        if which == "baseline":
                            seconds, emitted = baseline(target_model, prompt, args.tokens, seed)
                            row["baseline_seconds"] = seconds
                            row["baseline_per_token"] = seconds / emitted
                        else:
                            result = speculative(
                                target_model, draft_model, prompt, args.tokens, seed, arm
                            )
                            row |= result
                            row["speculative_seconds"] = result["seconds"]
                            row["speculative_per_token"] = result["seconds"] / result["emitted"]
                    row.pop("seconds", None)
                    runs.append(row)
                summary = {"regime": regime} | summarize(arm, prompt_name, runs, stats_rng)
                results.append(summary)
                raw.append(
                    {"regime": regime, "arm": arm["name"], "prompt": prompt_name, "runs": runs}
                )
                print(
                    f"{regime:<6} {arm['name']:<15} {prompt_name:<10} "
                    f"accept={summary['acceptance_rate_pooled'] or 0:.3f} "
                    f"speedup={summary['speedup_median']:.3f} "
                    f"[{summary['speedup_ci95'][0]:.3f}, {summary['speedup_ci95'][1]:.3f}] "
                    f"p={summary['sign_test_p']:.2g}",
                    flush=True,
                )
    return {
        "fixtures": fixtures,
        "entropy": entropy,
        "results": results,
        "raw": raw,
        "library": library,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--target-layers", type=int, default=8)
    parser.add_argument("--draft-layers", type=int, default=1)
    parser.add_argument("--fixtures", default="models/speculative")
    parser.add_argument("--serial", action="store_true", help="run regimes one after another")
    parser.add_argument("--output", default="bench/results/speculative-speedups.json")
    args = parser.parse_args()

    measured_at = datetime.now(timezone.utc).isoformat()
    jobs = [(regime, std, args, 264 + i) for i, (regime, std) in enumerate(REGIMES.items())]
    if args.serial:
        regimes = [run_regime(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            regimes = list(pool.map(run_regime, *zip(*jobs)))
    fixtures = {regime: r["fixtures"] for regime, r in zip(REGIMES, regimes)}
    entropy = {regime: r["entropy"] for regime, r in zip(REGIMES, regimes)}
    results = [row for r in regimes for row in r["results"]]
    raw = [row for r in regimes for row in r["raw"]]
    library = regimes[0]["library"]
    compiler = subprocess.run(
        ["g++", "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    output = {
        "task": "T264",
        "measured_at_utc": measured_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_processors": os.cpu_count(),
            "python": platform.python_version(),
            "compiler": compiler[0] if compiler else None,
            "library": library.replace("\\", "/"),
        },
        "fixtures": fixtures,
        "first_token_entropy_bits": entropy,
        "protocol": {
            "emitted_tokens": args.tokens,
            "seeds": args.seeds,
            "temperature": 1.0,
            "eos": "disabled so every run emits the same number of tokens",
            "timing": "setup prefill excluded; per-token time compared within each seed",
            "order": "baseline and speculative alternate which runs first by seed parity",
            "warmup_runs_discarded": "2 per regime",
            "concurrency": "serial"
            if args.serial
            else "the two regimes ran concurrently in separate single-threaded processes; "
            "arms are paired within each process",
            "speedup": "baseline seconds per token / speculative seconds per token",
            "ci95": "percentile bootstrap of the median over seeds, 10000 resamples, "
            "generator seeds 264 (peaked) and 265 (flat)",
            "sign_test": "exact two-sided binomial test of speedup > 1 across seeds",
        },
        "scope": "random-weight fixtures; Python orchestration over the native C ABI with "
        "serial target verification. Not trained-model acceptance, text quality or "
        "production speed.",
        "results": results,
        "raw": raw,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
