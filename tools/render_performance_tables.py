"""Render docs/performance.md from recorded bench/results artifacts (T266).

Every number in the generated page is read from a committed JSON result; nothing is typed
by hand. Run with --check to fail when the page is stale relative to the results.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/performance.md"


def load(name):
    return json.loads((ROOT / "bench/results" / name).read_text(encoding="utf-8"))


def ms(seconds):
    return f"{seconds * 1e3:.3f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def section(title, source, host, body, limits):
    return (
        f"## {title}\n\nSource: [{source}](../bench/results/{source}). Host: {host}.\n\n"
        f"{body}\n\nLimits: {limits}\n"
    )


def hosts():
    machine = load("machine.json")
    spec = load("speculative-speedups.json")["host"]
    memory = machine["physical_memory_bytes"] / 2**30
    rows = [
        [
            "A",
            machine["cpu"],
            machine["logical_processors"],
            f"{memory:.1f} GiB",
            machine["os"],
            machine["compiler"],
            machine["python"],
        ],
        [
            "B",
            spec["processor"],
            spec["logical_processors"],
            "not recorded",
            spec["platform"],
            spec["compiler"],
            spec["python"],
        ],
    ]
    return table(["Host", "CPU", "Logical CPUs", "RAM", "OS", "Compiler", "Python"], rows)


def matched_baseline():
    data = load("matched-baseline.json")
    rows = []
    for row in data["rows"]:
        for engine, label in (("tvara_native", "Tvara"), ("llama_cpp", "llama.cpp")):
            timings = row[engine]
            prefill = statistics.median(timings["prefill_seconds"])
            decode = statistics.median(timings["decode_seconds"])
            rows.append(
                [
                    row["quantization"].upper(),
                    label,
                    ms(prefill),
                    f"{len(data['prompt_tokens']) / prefill:,.0f}",
                    ms(decode),
                    f"{len(data['decode_tokens']) / decode:,.0f}",
                    f"{row['maximum_logit_error']:.2e}",
                ]
            )
    body = (
        f"{len(data['prompt_tokens'])} prompt tokens, {len(data['decode_tokens'])} decode tokens, "
        f"{data['threads']} thread, {data['warmup_runs']} warm-up and {data['measured_runs']} "
        f"measured runs; llama.cpp revision `{data['llama_cpp_revision'][:12]}`. Medians.\n\n"
        + table(
            [
                "Weights",
                "Engine",
                "Prefill ms",
                "Prefill tok/s",
                "Decode ms",
                "Decode tok/s",
                "Max logit error",
            ],
            rows,
        )
    )
    return section(
        "Matched prefill and decode baseline (T259)",
        "matched-baseline.json",
        "A (inferred from machine.json; the file records no host)",
        body,
        data["scope"] + ".",
    )


def repeated_native():
    data = load("repeated-native.json")["summary"]
    rows = [
        [
            phase,
            ms(data[key]["median"]),
            ms(data[key]["minimum"]),
            ms(data[key]["maximum"]),
            data[key]["samples"],
        ]
        for phase, key in (("Prefill", "prefill_seconds"), ("Decode", "decode_seconds"))
    ]
    full = load("repeated-native.json")
    return section(
        "Repeated native timings (T260)",
        "repeated-native.json",
        "A (inferred from machine.json; the file records no host)",
        table(["Phase", "Median ms", "Min ms", "Max ms", "Samples"], rows),
        full["scope"],
    )


def memory_bandwidth():
    data = load("memory-bandwidth.json")
    rows = [
        [kernel.title(), f"{r['best_mb_per_second']:,.1f}", f"{r['average_seconds'] * 1e3:.2f}"]
        for kernel, r in data["results"].items()
    ]
    body = f"{data['array_elements']:,} elements per array, {data['threads']} thread.\n\n" + table(
        ["Kernel", "Best MB/s", "Average ms"], rows
    )
    return section(
        "STREAM memory bandwidth (T258)",
        "memory-bandwidth.json",
        "A (inferred from machine.json; the file records no host)",
        body,
        data["scope"].rstrip(".") + ". Not checked against the STREAM run rules; a local run only.",
    )


def kernel_overhead():
    data = load("kernel-overhead.json")
    rows = [
        ["No-op thread pool dispatch", f"{data['noop_worker_us']:.1f} us", data["workers"]],
        ["Q8_0 32-element dot product", f"{data['q8_dot_ns']:.1f} ns", 1],
    ]
    return section(
        "Kernel overhead microbenchmark",
        "kernel-overhead.json",
        "A",
        table(["Operation", "Time per call", "Workers"], rows),
        data["limitation"] + ".",
    )


def prefix_cache():
    data = load("prefix-cache-traces.json")
    s = data["summary"]
    rows = [
        [
            s["turns"],
            f"{s['reused_token_fraction']:.1%}",
            f"{s['turn_hit_rate']:.0%}",
            f"{s['prefill_seconds_off_median']:.3f}",
            f"{s['prefill_seconds_on_median']:.3f}",
            f"{s['prefill_speedup_median']:.2f}x",
            s["outputs_equal"],
        ]
    ]
    radix = load("radix-prefix.json")
    body = (
        table(
            [
                "Turns",
                "Reused tokens",
                "Turn hits",
                "Prefill s (off)",
                "Prefill s (on)",
                "Speedup",
                "Outputs equal",
            ],
            rows,
        )
        + f"\n\nThe synthetic radix benchmark ([radix-prefix.json](../bench/results/radix-prefix.json)) "
        f"served {radix['requests']:,} requests with a {radix['hit_rate']:.0%} hit rate and "
        f"{radix['lookup_restore_ns_total'] / radix['requests']:.0f} ns mean lookup/restore."
    )
    return section(
        "Prefix reuse on conversation traces (T262)",
        "prefix-cache-traces.json",
        "A",
        body,
        data["workload"] + "; tiny random-weight model, synthetic token IDs, single thread.",
    )


def scheduler():
    data = load("scheduler-concurrency.json")
    rows = [
        [
            r["concurrency"],
            ms(r["ttft"]["p50_seconds"]),
            ms(r["ttft"]["p99_seconds"]),
            ms(r["inter_token"]["p50_seconds"]),
            ms(r["inter_token"]["p99_seconds"]),
            f"{r['throughput_tokens_per_second']['median']:,.0f}",
        ]
        for r in data["rows"]
    ]
    body = (
        f"{data['prompt_tokens']} prompt and {data['max_tokens']} output tokens per request, "
        f"{data['repetitions']} repetitions per level.\n\n"
        + table(
            [
                "Concurrency",
                "TTFT p50 ms",
                "TTFT p99 ms",
                "ITL p50 ms",
                "ITL p99 ms",
                "Throughput tok/s",
            ],
            rows,
        )
    )
    return section(
        "Scheduler latency under concurrency (T263)",
        "scheduler-concurrency.json",
        "A",
        body,
        data["fixture"] + "; " + data["backend"] + ". p99 rests on few samples.",
    )


def speculative():
    data = load("speculative-speedups.json")
    groups = {}
    for row in data["results"]:
        groups.setdefault((row["regime"], row["arm"]), []).append(row)
    rows = []
    for (regime, arm), items in groups.items():
        proposed = sum(r["proposed"] for r in items)
        accepted = sum(r["accepted"] for r in items)
        medians = [r["speedup_median"] for r in items]
        rows.append(
            [
                regime,
                arm,
                f"{accepted / proposed:.3f}" if proposed else "n/a",
                f"{min(medians):.2f}-{max(medians):.2f}",
                f"{min(r['speedup_ci95'][0] for r in items):.2f}-"
                f"{max(r['speedup_ci95'][1] for r in items):.2f}",
                f"{sum(r['seeds_faster'] for r in items)}/{sum(r['seeds'] for r in items)}",
                f"{max(r['sign_test_p'] for r in items):.2g}",
            ]
        )
    protocol = data["protocol"]
    target = data["fixtures"]["peaked"]["target"]
    body = (
        f"Target: {target['layers']} layers, width {target['hidden']}; draft: its first "
        f"{data['fixtures']['peaked']['draft']['layers']} layer. {protocol['seeds']} paired seeds "
        f"x {len({r['prompt'] for r in data['results']})} prompts per row, "
        f"{protocol['emitted_tokens']} tokens, temperature {protocol['temperature']}. Speedup "
        "columns give the range of per-prompt medians and the widest per-prompt bootstrap 95% "
        "interval; values below 1 are slowdowns.\n\n"
        + table(
            [
                "Regime",
                "Proposer",
                "Acceptance",
                "Speedup (median range)",
                "95% CI range",
                "Seeds faster",
                "Max sign-test p",
            ],
            rows,
        )
    )
    return section(
        "Speculative decoding speedups (T264)",
        "speculative-speedups.json",
        "B",
        body,
        data["scope"],
    )


def structured():
    data = load("structured-output-compliance.json")
    t = data["totals"]
    rows = [
        [
            r["schema"],
            r["requests"],
            r["stop"],
            r["valid"],
            r["length"],
            r["valid_prefix"],
            r["http_errors"],
        ]
        for r in data["schemas"]
    ]
    rows.append(
        [
            "**total**",
            t["requests"],
            t["stop"],
            t["valid"],
            t["length"],
            t["valid_prefix"],
            t["http_errors"],
        ]
    )
    rejected = data["unsupported_rejections"]
    body = (
        table(
            [
                "Schema",
                "Requests",
                "Finished",
                "Schema-valid",
                "Length-limited",
                "Valid prefix",
                "HTTP errors",
            ],
            rows,
        )
        + f"\n\nStreaming matched non-streaming output in {t['stream_checked'] - t['stream_mismatch']}"
        f"/{t['stream_checked']} checks. {sum(r['rejected'] for r in rejected)}/{len(rejected)} "
        f"unsupported schemas returned HTTP 400. Status: **{data['status']}**."
    )
    return section(
        "Structured-output compliance (T265)",
        "structured-output-compliance.json",
        "B",
        body,
        data["scope"] + ". Finish rates depend on the random model emitting whitespace.",
    )


def grammar():
    masks = load("grammar-masks.json")
    compare = load("grammar-mask-comparison.json")
    rows = [
        [
            "Schema mask, 256-token vocabulary",
            f"{masks['latency_ns']['median'] / 1e6:.2f}",
            f"{masks['latency_ns']['p95'] / 1e6:.2f}",
            masks["latency_ns"]["samples"],
        ]
    ] + [
        [
            f"{name.replace('_', ' ')}, {compare['vocabulary_size']:,}-token vocabulary",
            f"{value / 1e6:.2f}",
            "",
            compare["repetitions"],
        ]
        for name, value in compare["median_ns"].items()
    ]
    return section(
        "Grammar mask cost",
        "grammar-masks.json",
        "A",
        table(["Path", "Median ms", "p95 ms", "Samples"], rows),
        masks["scope"] + "; " + compare["scope"] + ".",
    )


def quantized_kv():
    data = load("variants/quantized-kv.json")
    rows = [
        [
            m["context"],
            f"{m['memory_ratio']:.3f}",
            f"{m['maximum_kv_error']:.4f}",
            f"{m['attention_max_error']:.4f}",
            m["projected_argmax_agreement"],
        ]
        for m in data["measurements"]
    ]
    return section(
        "Quantized KV prototype",
        "variants/quantized-kv.json",
        "A",
        table(
            ["Context", "Int8/float bytes", "Max KV error", "Max attention error", "Argmax agrees"],
            rows,
        ),
        data["scope"] + "; prototype not used by the served executor.",
    )


def regression():
    data = load("final-regression.json")
    rows = [
        [
            " ".join(map(str, r["tokens"])),
            f"{r['comparison']['max_absolute_error']:.2e}",
            f"{r['native_perplexity']:.3f}",
            f"{r['oracle_perplexity']:.3f}",
        ]
        for r in data["logits"]
    ]
    return section(
        "Correctness regression (T261)",
        "final-regression.json",
        "A",
        table(["Tokens", "Max logit error", "Native perplexity", "Oracle perplexity"], rows)
        + f"\n\nGreedy tokens match the oracle: {data['greedy_matches']}.",
        data["scope"] + ".",
    )


def speculative_limit():
    results = load("speculative-speedups.json")["results"]
    faster = [r for r in results if r["speedup_ci95"][0] > 1]
    outcome = (
        "it was slower than plain decoding in every measured row"
        if all(r["speedup_ci95"][1] < 1 for r in results)
        else f"{len(faster)} of {len(results)} measured rows were reliably faster"
    )
    return (
        "- Speculative verification runs target steps serially and rollback re-decodes the "
        f"last kept token; {outcome}.\n"
    )


def render():
    parts = [
        "# Performance results\n",
        "Generated by `python tools/render_performance_tables.py` from the JSON artifacts in "
        "`bench/results/`; do not edit by hand. Every model is a seeded random-weight fixture, "
        "so no table describes pretrained-model quality, useful text, or the speed of a "
        "0.5-3B model. Measurements come from two machines and are not comparable across "
        "hosts. Measurement protocol: [benchmarks.md](benchmarks.md).\n",
        "## Hosts\n\n" + hosts() + "\n\nNeither host controlled CPU frequency, temperature or "
        "background load. Results without a per-run host field are attributed to host A "
        "because they were recorded in the same session as `machine.json`; this is inferred.\n",
        regression(),
        matched_baseline(),
        repeated_native(),
        memory_bandwidth(),
        kernel_overhead(),
        prefix_cache(),
        scheduler(),
        speculative(),
        structured(),
        grammar(),
        quantized_kv(),
        "## Known performance limitations\n\n"
        "- The served executor dequantizes all weights to float32 at load and runs scalar "
        "matvec on one thread; quantized and SIMD kernels are not on the model path "
        "([kernels.md](kernels.md)).\n"
        "- Prompt prefill is one C ABI call per token; there is no batched prompt matmul.\n"
        "- The HTTP server does not use the scheduler, prefix cache, or speculative decoding.\n"
        + speculative_limit()
        + "- AVX-512 and ARM NEON paths have not run on supporting hardware "
        "([avx512-validation.md](avx512-validation.md), [arm-validation.md](arm-validation.md)).\n"
        "- No pretrained model has been benchmarked; tiny fixtures exaggerate per-call overhead.\n",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the page is stale")
    args = parser.parse_args()
    text = render()
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            sys.exit("docs/performance.md is stale; run tools/render_performance_tables.py")
        return
    OUTPUT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
