"""Build portable or native Yase distributions through the PEP 517 frontend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        action="store_true",
        help="include the ABI-specific C++ extension in the wheel",
    )
    parser.add_argument(
        "--sdist-only",
        action="store_true",
        help="build only the source distribution",
    )
    args = parser.parse_args()

    environment = os.environ.copy()
    if args.native:
        environment["YASE_BUILD_NATIVE"] = "1"
    command = [sys.executable, "-m", "build", "--sdist"]
    if not args.sdist_only:
        command.append("--wheel")
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
