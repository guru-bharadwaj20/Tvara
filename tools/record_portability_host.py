"""Record actual CI execution context without implying controlled hardware timing."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-details", required=True)
    args = parser.parse_args()
    result = dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_details_file=args.cpu_details,
        execution_environment="GitHub-hosted virtual machine"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else "local execution; no hosted runner assumed",
        conditions="CPU affinity, frequency, thermal state and concurrent/shared host load were not controlled; synthetic kernel timing is not an end-to-end speed claim",
        workflow={
            key: os.environ.get(key)
            for key in (
                "GITHUB_REPOSITORY",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_SHA",
                "RUNNER_OS",
                "RUNNER_ARCH",
            )
        },
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
