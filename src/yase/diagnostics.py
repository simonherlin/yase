"""Dependency-light runtime health and readiness diagnostics."""

import importlib.util
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class RuntimeInfo:
    """Environment facts useful for reproducible support reports."""

    python: str
    platform: str
    machine: str
    numpy: str
    cpu_count: Optional[int]
    optional_packages: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "platform": self.platform,
            "machine": self.machine,
            "numpy": self.numpy,
            "cpu_count": self.cpu_count,
            "optional_packages": dict(self.optional_packages),
        }


@dataclass(frozen=True)
class HealthReport:
    """Readiness result with machine-readable checks and human detail."""

    status: str
    checks: Mapping[str, bool]
    details: Mapping[str, str] = field(default_factory=dict)
    runtime: Optional[RuntimeInfo] = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "degraded", "unhealthy"):
            raise ValueError("status must be ok, degraded, or unhealthy")

    @property
    def ready(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "ready": self.ready,
            "checks": dict(self.checks),
            "details": dict(self.details),
            "runtime": None if self.runtime is None else self.runtime.to_dict(),
        }


def collect_runtime_info(
    optional_packages: tuple[str, ...] = (
        "cv2",
        "onnxruntime",
        "openvino",
        "tensorrt",
        "torch",
        "transformers",
        "paddleocr",
        "yase._native",
    ),
) -> RuntimeInfo:
    """Inspect installed capabilities without importing heavy packages."""
    return RuntimeInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        numpy=np.__version__,
        cpu_count=os.cpu_count(),
        optional_packages={
            package: importlib.util.find_spec(package) is not None
            for package in optional_packages
        },
    )


def health_check(
    extractor: Optional[Any] = None,
    *,
    required_packages: tuple[str, ...] = (),
) -> HealthReport:
    """Return readiness without downloading weights or invoking inference."""
    checks = {"numpy": True}
    details: dict[str, str] = {}
    runtime = collect_runtime_info()
    for package in required_packages:
        available = importlib.util.find_spec(package) is not None
        checks[f"package:{package}"] = available
        if not available:
            details[f"package:{package}"] = "not installed"

    if extractor is not None:
        usable = any(
            hasattr(extractor, name) for name in ("extract", "predict", "run")
        ) or callable(extractor)
        checks["extractor_interface"] = usable
        if not usable:
            details["extractor_interface"] = (
                "expected extract, predict, run, or callable"
            )

    status = "ok" if all(checks.values()) else "degraded"
    return HealthReport(status=status, checks=checks, details=details, runtime=runtime)


__all__ = ["HealthReport", "RuntimeInfo", "collect_runtime_info", "health_check"]
