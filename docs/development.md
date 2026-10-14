# Local development

Requirements: Python 3.11+, C++20 compiler, CMake 3.24+, and a Make or Ninja generator.
From the repository root, create a virtual environment and install the package:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/cmake.exe --preset debug -G "MinGW Makefiles"
.venv/Scripts/cmake.exe --build --preset debug
.venv/Scripts/ctest.exe --preset debug
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check python tests/python tools
.venv/Scripts/ruff.exe format --check python tests/python tools
```

For release, substitute the release preset in the configure, build, and test steps.
On Linux activate the environment, omit the Windows executable paths, and omit the
MinGW generator. Add -DTVARA_SANITIZERS=ON to a Linux debug configuration.

Locally exercised: Windows x86-64, GCC 15.2, Python 3.13, CMake 3.31.6, debug and
release tensor tests, and Python smoke checks. Linux sanitizer execution is a CI
responsibility and must not be inferred from these Windows results.
