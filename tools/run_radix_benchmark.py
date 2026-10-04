import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

result = json.loads(subprocess.check_output(["build/radix_bench.exe"], text=True))
result["hit_rate"] = result["hits"] / result["requests"]
result["saved_token_fraction"] = result["saved_prefill_tokens"] / result["requested_tokens"]
result.update(
    measured_at_utc=datetime.now(timezone.utc).isoformat(),
    platform=platform.platform(),
    workload="synthetic KV, 75% exact repeats, 25% disjoint; measures lookup and reference restoration, not model prefill",
)
Path("bench/results").mkdir(parents=True, exist_ok=True)
Path("bench/results/radix-prefix.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
assert result["hits"] == 7500 and result["saved_prefill_tokens"] == 60000
