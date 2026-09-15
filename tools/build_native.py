"""Build the optional C++ acceleration module in-place."""

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path


def _python_include_dirs() -> list[Path]:
    configured = os.environ.get("YASE_PYTHON_INCLUDE_DIR")
    include = Path(configured) if configured else Path(sysconfig.get_path("include"))
    candidates = [include, include.parent]
    unique: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in unique:
            unique.append(candidate)
    if not (include / "Python.h").is_file():
        raise SystemExit(
            "Python development headers are required to build Yase native "
            f"acceleration; Python.h was not found in {include}. Install the "
            "matching python-dev package or set YASE_PYTHON_INCLUDE_DIR."
        )
    return unique


def main() -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("a C++17 compiler (g++ or clang++) is required")
    root = Path(__file__).resolve().parents[1]
    source = root / "native" / "yase_native.cpp"
    output = root / "src" / "yase" / f"_native{sysconfig.get_config_var('EXT_SUFFIX')}"
    include_dirs = _python_include_dirs()
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *(f"-I{directory}" for directory in include_dirs),
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
