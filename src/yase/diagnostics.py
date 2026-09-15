"""Dependency-light runtime health and readiness diagnostics."""

import importlib.util
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuntimeInfo:
    """Environment facts useful for reproducible support reports."""

    python: str
    platform: str
    machine: str
    numpy: str
    cpu_count: int | None
    optional_packages: Mapping[str, bool] = field(default_factory=dict)
    provider_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "platform": self.platform,
            "machine": self.machine,
            "numpy": self.numpy,
            "cpu_count": self.cpu_count,
            "optional_packages": dict(self.optional_packages),
            "provider_info": dict(self.provider_info),
        }


@dataclass(frozen=True)
class HealthReport:
    """Readiness result with machine-readable checks and human detail."""

    status: str
    checks: Mapping[str, bool]
    details: Mapping[str, str] = field(default_factory=dict)
    runtime: RuntimeInfo | None = None

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


def collect_provider_info() -> dict[str, Any]:
    """Probe installed inference providers on explicit request.

    The function is intentionally separate from :func:`collect_runtime_info`:
    importing or initializing accelerator runtimes can be expensive and may
    touch drivers. Every provider is optional and failures are represented as
    data instead of making a diagnostic command crash.
    """
    info: dict[str, Any] = {}

    try:
        import onnxruntime as ort

        info["onnxruntime"] = {
            "version": str(getattr(ort, "__version__", "unknown")),
            "available_providers": list(ort.get_available_providers()),
        }
    except Exception as exc:  # pragma: no cover - provider installation varies
        info["onnxruntime"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        import openvino as ov

        core = ov.Core()
        devices = []
        for device in core.available_devices:
            item: dict[str, Any] = {"device": str(device)}
            try:
                item["full_name"] = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception as exc:  # pragma: no cover - device plugin varies
                item["property_error"] = f"{type(exc).__name__}: {exc}"
            devices.append(item)
        info["openvino"] = {
            "version": str(ov.get_version()),
            "devices": devices,
        }
    except Exception as exc:  # pragma: no cover - provider installation varies
        info["openvino"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        cuda_info: dict[str, Any] = {"available": cuda}
        if cuda:
            cuda_info["device_count"] = int(torch.cuda.device_count())
            cuda_info["devices"] = [
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            ]
        info["torch"] = {
            "version": str(getattr(torch, "__version__", "unknown")),
            "cuda": cuda_info,
        }
    except Exception as exc:  # pragma: no cover - provider installation varies
        info["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        import tensorrt as trt

        info["tensorrt"] = {
            "version": str(getattr(trt, "__version__", "unknown")),
            "available": True,
        }
    except Exception as exc:  # pragma: no cover - provider installation varies
        info["tensorrt"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return info


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
    *,
    probe_providers: bool = False,
) -> RuntimeInfo:
    """Inspect installed capabilities without importing heavy packages.

    Set ``probe_providers=True`` only for an explicit runtime readiness probe.
    """
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
        provider_info=collect_provider_info() if probe_providers else {},
    )


def health_check(
    extractor: Any | None = None,
    *,
    required_packages: tuple[str, ...] = (),
    probe_providers: bool = False,
) -> HealthReport:
    """Return readiness without downloading weights or invoking inference."""
    checks = {"numpy": True}
    details: dict[str, str] = {}
    runtime = collect_runtime_info(probe_providers=probe_providers)
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


__all__ = [
    "HealthReport",
    "RuntimeInfo",
    "collect_provider_info",
    "collect_runtime_info",
    "health_check",
]
