import re
import tomllib
from pathlib import Path

from tvara import __version__

ROOT = Path(__file__).resolve().parents[1].parent


def test_release_version_is_consistent_across_build_files():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    cmake = re.search(r"project\(tvara VERSION (\S+)", (ROOT / "CMakeLists.txt").read_text())
    native = re.search(r'return "([^"]+)"', (ROOT / "core/src/version.cpp").read_text())
    assert __version__ == project["version"] == cmake.group(1) == native.group(1) == "0.1.0"
