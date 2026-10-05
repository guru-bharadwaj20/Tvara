"""Reproducible tiny/random native scheduling experiment."""

import copy
import datetime
import json
from pathlib import Path
import time
from tvara.native import NativeModel
from tvara.scheduler import Scheduler, NativeSchedulerBackend, conversation_load, latency_report


def measure(model_path="models/api-fixture/model.gguf", count=8):
    base = conversation_load(count, seed=19)
    rows = []
    outputs = []
    for mode in ("static", "continuous"):
        requests = copy.deepcopy(base)
        with NativeModel(model_path) as model:
            scheduler = Scheduler(NativeSchedulerBackend(model), max_active=4, prefill_chunk=4)
            start = time.perf_counter()
            for r in requests:
                r.submitted = start
            if mode == "continuous":
                for r in requests:
                    scheduler.submit(r)
                scheduler.run()
            else:
                for offset in range(0, len(requests), 4):
                    for r in requests[offset : offset + 4]:
                        scheduler.submit(r)
                    scheduler.run()
            elapsed = time.perf_counter() - start
        outputs.append([r.output for r in requests])
        rows.append(
            {
                "mode": mode,
                "seconds": elapsed,
                "tokens_per_second": sum(len(r.output) for r in requests) / elapsed,
                **latency_report(requests),
            }
        )
    if outputs[0] != outputs[1]:
        raise RuntimeError("Scheduling changed greedy outputs")
    result = {
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fixture": "random 1-layer native GGUF, not pretrained",
        "backend": "serial decode_batch, no SIMD batch inference",
        "seed": 19,
        "requests": count,
        "repetitions": 1,
        "rows": rows,
        "outputs_identical": True,
    }
    path = Path("bench/results/scheduler.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    measure()
