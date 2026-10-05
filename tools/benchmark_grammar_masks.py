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


def compare():
    from tvara.grammar import Grammar, GrammarState
    from tvara.token_masks import VocabularyTrie, ByteState, trie_allowed, naive_allowed, MaskEngine

    g = Grammar.from_text('root ::= "abc" [a-z]*')
    vocabulary = [
        ("abc" + "".join(chr(97 + (i // (26**power)) % 26) for power in (2, 1, 0))).encode()
        for i in range(2048)
    ] + [
        ("zzz" + "".join(chr(97 + (i // (26**power)) % 26) for power in (2, 1, 0))).encode()
        for i in range(2048)
    ]
    trie = VocabularyTrie(vocabulary)
    state = ByteState(GrammarState.start(g))
    engine = MaskEngine(vocabulary, g)
    samples = {"trie": [], "naive_scan": [], "cached": []}
    expected = naive_allowed(vocabulary, g, state)
    assert trie_allowed(trie, g, state) == expected
    for _ in range(15):
        for name, fn in (
            ("trie", lambda: trie_allowed(trie, g, state)),
            ("naive_scan", lambda: naive_allowed(vocabulary, g, state)),
            ("cached", lambda: engine.mask(state).ids()),
        ):
            begin = time.perf_counter_ns()
            result = fn()
            samples[name].append(time.perf_counter_ns() - begin)
            assert result == expected
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "vocabulary_size": len(vocabulary),
        "allowed_tokens": len(expected),
        "median_ns": {name: statistics.median(values) for name, values in samples.items()},
        "repetitions": 15,
        "cache_hits": engine.hits,
        "scope": "Synthetic shared-prefix vocabulary and one automaton state; no full-model throughput claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench/results/grammar-masks.json")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    result = compare() if args.compare else measure()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key in {"valid_outputs", "latency_ns", "median_ns"}
            }
        )
    )


if __name__ == "__main__":
    main()
