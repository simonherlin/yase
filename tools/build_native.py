"""Build the optional C++ acceleration module in-place."""

import shutil
import subprocess
import sysconfig
from pathlib import Path


def main() -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("a C++17 compiler (g++ or clang++) is required")
    root = Path(__file__).resolve().parents[1]
    source = root / "native" / "yase_native.cpp"
    output = root / "src" / "yase" / f"_native{sysconfig.get_config_var('EXT_SUFFIX')}"
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        f"-I{sysconfig.get_path('include')}",
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
