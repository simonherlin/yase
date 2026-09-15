"""Validate the contents of locally built Yase distributions."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def _latest(paths: list[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {description} found")
    return max(paths, key=lambda path: path.stat().st_mtime)


def _check_wheel(path: Path, *, native: bool) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    native_members = [name for name in names if "yase/_native" in name]
    if "yase/__init__.py" not in names:
        raise ValueError(f"{path} does not contain yase/__init__.py")
    if native and not native_members:
        raise ValueError(f"{path} does not contain the native extension")
    if not native and (not path.name.endswith("py3-none-any.whl") or native_members):
        raise ValueError(f"{path} is not a portable pure-Python wheel")


def _check_sdist(path: Path) -> None:
    required = {
        "pyproject.toml",
        "native/yase_native.cpp",
        "tools/build_package.py",
        "tools/build_wheels.py",
    }
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name.split("/", 1)[-1] for member in archive.getmembers()}
    missing = sorted(required - members)
    if missing:
        raise ValueError(f"{path} is missing sdist members: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--native", action="store_true", help="validate the latest native wheel"
    )
    args = parser.parse_args(argv)
    wheel_pattern = "yase-*.whl"
    wheels = list(args.dist_dir.glob(wheel_pattern))
    if args.native:
        wheels = [path for path in wheels if not path.name.endswith("py3-none-any.whl")]
    else:
        wheels = [path for path in wheels if path.name.endswith("py3-none-any.whl")]
    wheel = _latest(wheels, "native wheel" if args.native else "portable wheel")
    sdist = _latest(list(args.dist_dir.glob("yase-*.tar.gz")), "source archive")
    _check_wheel(wheel, native=args.native)
    _check_sdist(sdist)
    print(f"validated wheel: {wheel}")
    print(f"validated sdist: {sdist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
