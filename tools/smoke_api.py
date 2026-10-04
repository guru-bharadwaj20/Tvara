"""Start a hidden local CPU server and verify HTTP/SSE with its random-weight fixture."""

import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx


def smoke():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "tvara.server",
        "--model",
        "models/api-fixture/model.gguf",
        "--tokenizer",
        "models/api-fixture/tokenizer.json",
        "--name",
        "native-fixture",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=10) as client:
            deadline = time.monotonic() + 20
            while True:
                try:
                    if client.get(base + "/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read().decode(errors="replace"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Local server did not become ready")
                time.sleep(0.1)
            body = {"model": "native-fixture", "prompt": "hello", "max_tokens": 3}
            response = client.post(base + "/v1/completions", json=body)
            response.raise_for_status()
            data = response.json()
            assert data["object"] == "text_completion" and data["usage"]["prompt_tokens"] == 5
            stream = client.post(base + "/v1/completions", json={**body, "stream": True})
            stream.raise_for_status()
            assert stream.text.endswith("data: [DONE]\n\n")
            result = {
                "executed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "backend": "real C++ CPU executor",
                "model": "seeded random-weight 257-token fixture; not pretrained",
                "http_status": response.status_code,
                "sse_done": True,
                "completion_tokens": data["usage"]["completion_tokens"],
            }
            output = Path("bench/results/api-smoke.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stderr:
            process.stderr.close()


if __name__ == "__main__":
    smoke()
