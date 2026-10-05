import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tvara.native import NativeModel
from tvara.speculative import TokenizerIdentity, NativeSpeculativeModel, speculative_step


class BenchModel:
    identity = TokenizerIdentity((b"a", b"b", b"c"), "synthetic-benchmark-v1")

    def __init__(self, p):
        self.p, self.history = np.asarray(p), [0]

    def distribution(self):
        return self.p.copy()

    def advance(self, token):
        self.history.append(token)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, position):
        del self.history[position:]


records = []
for name, p, q in [
    ("high-overlap", [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
    ("moderate-overlap", [0.2, 0.3, 0.5], [0.5, 0.3, 0.2]),
    ("disjoint", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
]:
    rng = np.random.default_rng(991)
    draft, target = BenchModel(q), BenchModel(p)
    accepted = proposed = emitted = 0
    start = time.perf_counter()
    for _ in range(2000):
        result = speculative_step(draft, target, 4, rng)
        accepted += result.accepted
        proposed += result.proposed
        emitted += len(result.tokens)
    elapsed = time.perf_counter() - start
    baseline_start = time.perf_counter()
    baseline_rng = np.random.default_rng(991)
    for _ in range(emitted):
        baseline_rng.choice(3, p=p)
    baseline_seconds = time.perf_counter() - baseline_start
    records.append(
        {
            "workload": name,
            "proposals": proposed,
            "accepted": accepted,
            "acceptance_rate": accepted / proposed,
            "emitted": emitted,
            "speculative_seconds": elapsed,
            "baseline_seconds": baseline_seconds,
            "speedup": baseline_seconds / elapsed,
        }
    )
Path("bench/results").mkdir(parents=True, exist_ok=True)
Path("bench/results/speculative-overhead.json").write_text(
    json.dumps(
        {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "synthetic constant-probability model; Python orchestration overhead only, no natural-language/code/JSON inference performance claim",
            "cases": records,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


identity = TokenizerIdentity(
    tuple(bytes([i]) for i in range(256)) + (b"",), "byte-fixture-v1", None
)
root = Path.cwd()
libraries = [
    root / "build/debug" / name
    for name in ("libtvara_capi.dll", "tvara_capi.dll", "libtvara_capi.so", "libtvara_capi.dylib")
]
library = next(p for p in libraries if p.exists())
native_records = []
for label, text in [
    ("text-shaped", "Tell me"),
    ("code-shaped", "def x():"),
    ("JSON-shaped", '{"x":'),
]:
    prompt = list(text.encode("utf-8"))
    with (
        NativeModel(root / "models/api-fixture/model.gguf", library=library) as draft_model,
        NativeModel(root / "models/api-fixture/model.gguf", library=library) as target_model,
    ):
        baseline = NativeSpeculativeModel(target_model, prompt, identity)
        baseline_rng = np.random.default_rng(991)
        try:
            start = time.perf_counter()
            for _ in range(8):
                token = int(baseline_rng.choice(len(identity.pieces), p=baseline.distribution()))
                baseline.advance(token)
            baseline_seconds = time.perf_counter() - start
        finally:
            baseline.close()
        draft = NativeSpeculativeModel(draft_model, prompt, identity, temperature=0.5)
        target = NativeSpeculativeModel(target_model, prompt, identity)
        try:
            rng = np.random.default_rng(991)
            start = time.perf_counter()
            emitted = accepted = proposed = 0
            output = []
            while len(output) < 8:
                result = speculative_step(draft, target, min(2, 7 - len(output)), rng)
                proposed += result.proposed
                accepted += result.accepted
                output.extend(result.tokens)
                if output[-1] == identity.eos_id:
                    break
            elapsed = time.perf_counter() - start
            native_records.append(
                {
                    "prompt_shape": label,
                    "prompt_tokens": len(prompt),
                    "emitted_tokens": len(output),
                    "proposed": proposed,
                    "accepted": accepted,
                    "acceptance_rate": accepted / proposed if proposed else None,
                    "speculative_seconds": elapsed,
                    "baseline_seconds": baseline_seconds,
                    "speedup": baseline_seconds / elapsed,
                }
            )
        finally:
            draft.close()
            target.close()
Path("bench/results/speculative-native.json").write_text(
    json.dumps(
        {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "real native random-weight 257-vocabulary fixture; diagnostic prompt shapes only, not trained text/code/JSON quality or production speed. Both timings exclude setup prefill and decode exactly eight tokens; both consume each emitted token to leave synchronized KV. Debug native build, single run; timings are diagnostic.",
            "cases": native_records,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
