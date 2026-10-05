"""Record public benchmark configuration without device serials or user paths."""

from datetime import datetime, timezone
import ctypes
import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    memory = None
    if os.name == "nt":

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64)
                for name in (
                    "total_phys",
                    "available_phys",
                    "total_page",
                    "available_page",
                    "total_virtual",
                    "available_virtual",
                    "extended",
                )
            ]

        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = status.total_phys
    elif hasattr(os, "sysconf"):
        memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_processors": os.cpu_count(),
        "physical_memory_bytes": memory,
        "compiler": compiler,
        "python": platform.python_version(),
        "native_build": "Release C++20",
        "limitations": "CPU frequency, temperature and background load are not controlled; native benchmarks identify worker count individually",
    }
    path = Path("bench/results/machine.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
