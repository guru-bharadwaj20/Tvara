"""Measure prefix reuse on seeded multi-turn conversation traces with the native engine.

Cache off: every turn prefills its full prompt in a fresh native sequence.
Cache on: the live native sequence from the previous turn of the same conversation is
retained, and only the new suffix (last reply token plus the new user turn) is decoded.
The native C API exposes no fork/truncate, so only exact whole-history prefixes within
one conversation are reusable; a shared system prompt is not shared across conversations.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tvara.native import NativeModel


def build_traces(seed, conversations, turns, vocab):
    rng = np.random.default_rng(seed)
    system = rng.integers(1, vocab, 64).tolist()
    return [
        {
            "system": system,
            "users": [
                rng.integers(1, vocab, int(rng.integers(16, 49))).tolist() for _ in range(turns)
            ],
        }
        for _ in range(conversations)
    ]


def prefill(model, sequence, tokens):
    start = time.perf_counter()
    for token in tokens:
        logits = model.logits(sequence, token)
    return logits, time.perf_counter() - start


def decode(model, sequence, logits, count):
    reply = []
    for index in range(count):
        token = int(logits.argmax())
        reply.append(token)
        if index + 1 < count:
            logits = model.logits(sequence, token)
    return reply


def run_trace(model, trace, reply_tokens, cached):
    history, turns, sequence, fed = list(trace["system"]), [], None, []
    try:
        for user in trace["users"]:
            history += user
            if cached and sequence is not None and history[: len(fed)] == fed:
                suffix = history[len(fed) :]
                reused = len(fed)
            else:
                if sequence is not None:
                    model._drop_sequence(sequence)
                sequence, fed, suffix, reused = model._new_sequence(), [], list(history), 0
            logits, seconds = prefill(model, sequence, suffix)
            fed += suffix
            reply = decode(model, sequence, logits, reply_tokens)
            fed += reply[:-1]
            history += reply
            turns.append(
                {
                    "prompt_tokens": len(history) - len(reply),
                    "reused_tokens": reused,
                    "prefill_seconds": seconds,
                    "logits_head_sum": float(np.float64(logits[:64]).sum()),
                    "reply": reply,
                }
            )
            if not cached:
                model._drop_sequence(sequence)
                sequence, fed = None, []
    finally:
        if sequence is not None:
            model._drop_sequence(sequence)
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/tiny-llama/reference.gguf")
    parser.add_argument("--seed", type=int, default=262)
    parser.add_argument("--conversations", type=int, default=4)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--reply-tokens", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="bench/results/prefix-cache-traces.json")
    args = parser.parse_args()
    model = NativeModel(args.model, max_blocks=256, block_tokens=16)
    traces = build_traces(args.seed, args.conversations, args.turns, model.vocab_size)
    runs = []
    for repeat in range(args.repeats):
        for mode in ("off", "on") if repeat % 2 == 0 else ("on", "off"):
            start = time.perf_counter()
            result = [run_trace(model, t, args.reply_tokens, mode == "on") for t in traces]
            runs.append(
                {
                    "repeat": repeat,
                    "mode": mode,
                    "wall_seconds": time.perf_counter() - start,
                    "traces": result,
                }
            )
    model.close()

    def turns_of(mode):
        return [
            [turn for trace in run["traces"] for turn in trace]
            for run in runs
            if run["mode"] == mode
        ]

    off, on = turns_of("off"), turns_of("on")
    equal = all(
        a["reply"] == b["reply"] and a["logits_head_sum"] == b["logits_head_sum"]
        for run_on, run_off in zip(on, off)
        for a, b in zip(run_on, run_off)
    )
    prompt_tokens = sum(t["prompt_tokens"] for t in on[0])
    reused = sum(t["reused_tokens"] for t in on[0])
    hits = sum(t["reused_tokens"] > 0 for t in on[0])
    off_prefill = [sum(t["prefill_seconds"] for t in run) for run in off]
    on_prefill = [sum(t["prefill_seconds"] for t in run) for run in on]
    summary = {
        "turns": len(on[0]),
        "prompt_tokens": prompt_tokens,
        "reused_prompt_tokens": reused,
        "reused_token_fraction": reused / prompt_tokens,
        "turn_hits": hits,
        "turn_hit_rate": hits / len(on[0]),
        "prefill_seconds_off_median": float(np.median(off_prefill)),
        "prefill_seconds_on_median": float(np.median(on_prefill)),
        "prefill_seconds_saved_median": float(np.median(off_prefill) - np.median(on_prefill)),
        "prefill_speedup_median": float(np.median(off_prefill) / np.median(on_prefill)),
        "outputs_equal": equal,
    }
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "model": args.model,
        "config": vars(args),
        "workload": (
            "seeded random token IDs; 64-token system prompt, 16-48 token user turns, greedy "
            "replies; cache on retains each conversation's live native sequence between turns"
        ),
        "summary": summary,
        "raw_prefill_seconds": {"off": off_prefill, "on": on_prefill},
        "runs": runs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not equal:
        raise SystemExit("cache on/off outputs differ")


if __name__ == "__main__":
    main()
