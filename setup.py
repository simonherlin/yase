"""Setuptools hook for the optional Yase native extension.

The extension is intentionally opt-in for normal source and wheel builds. A
portable pure-Python wheel keeps the base package installable without a C++
compiler, while CI and cibuildwheel can request an ABI-specific wheel with
``YASE_BUILD_NATIVE=1``.
"""

from __future__ import annotations

import os
import sys

from setuptools import Extension, setup


def _native_requested() -> bool:
    value = os.environ.get("YASE_BUILD_NATIVE", "")
    return value.lower() in {"1", "true", "yes", "on"} or bool(
        os.environ.get("CIBUILDWHEEL")
    )


def _native_extensions() -> list[Extension]:
    if not _native_requested():
        return []
    compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
    return [
        Extension(
            "yase._native",
            sources=["native/yase_native.cpp"],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]


setup(ext_modules=_native_extensions())
