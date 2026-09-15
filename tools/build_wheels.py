"""Build Yase native wheels with cibuildwheel in a release environment.

This wrapper intentionally does not install dependencies or create CI
configuration. Install cibuildwheel in a dedicated environment, then select
the platform and identifiers supported by the host/toolchain.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

SUPPORTED_PYTHONS = "cp310-* cp311-* cp312-* cp313-*"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("auto", "linux", "macos", "windows"),
        default="auto",
        help="cibuildwheel target platform (default: auto)",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="IDENTIFIER",
        help="build only one cibuildwheel identifier; repeatable",
    )
    parser.add_argument(
        "--archs",
        help="optional cibuildwheel architecture selector, e.g. native or auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wheelhouse"),
        help="directory for produced wheels",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="pass through to cibuildwheel for local debugging only",
    )
    args = parser.parse_args(argv)

    executable = shutil.which("cibuildwheel")
    if executable is None:
        executable = shutil.which("uvx")
        if executable is None:
            parser.error(
                "cibuildwheel is not installed; install it in the release "
                "environment or install uv to use uvx"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("CIBW_BUILD", SUPPORTED_PYTHONS)
    environment.setdefault("CIBW_ENVIRONMENT", "YASE_BUILD_NATIVE=1")
    command = [executable]
    if Path(executable).name == "uvx":
        command.append("cibuildwheel")
    command.extend(["--output-dir", str(args.output_dir)])
    if args.platform != "auto":
        command.extend(["--platform", args.platform])
    if args.only:
        environment["CIBW_BUILD"] = " ".join(args.only)
    if args.archs:
        command.extend(["--archs", args.archs])
    if args.no_isolation:
        command.append("--no-isolation")
    command.append(".")
    print("$", " ".join(command))
    subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
