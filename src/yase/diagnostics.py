"""Dependency-light runtime health and readiness diagnostics."""

import importlib.util
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np

_PACKAGE_IMPORT_ALIASES = {
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "onnxruntime-gpu": "onnxruntime",
    "opentelemetry-api": "opentelemetry",
    "paddlepaddle": "paddle",
    "paddlepaddle-gpu": "paddle",
    "qdrant-client": "qdrant_client",
}


def _package_import_name(package: str) -> str:
    """Map a distribution name accepted by the CLI to its import name."""
    normalized = package.strip().lower()
    return _PACKAGE_IMPORT_ALIASES.get(normalized, package)


def _package_is_available(package: str) -> bool:
    return importlib.util.find_spec(_package_import_name(package)) is not None


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
    optional_versions: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "platform": self.platform,
            "machine": self.machine,
            "numpy": self.numpy,
            "cpu_count": self.cpu_count,
            "optional_packages": dict(self.optional_packages),
            "provider_info": dict(self.provider_info),
            "optional_versions": dict(self.optional_versions),
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
            device_count = int(torch.cuda.device_count())
            cuda_info["device_count"] = device_count
            cuda_info["devices"] = [
                str(torch.cuda.get_device_name(index)) for index in range(device_count)
            ]
            get_capability = getattr(torch.cuda, "get_device_capability", None)
            capabilities = []
            if callable(get_capability):
                capabilities = [
                    list(get_capability(index)) for index in range(device_count)
                ]
                cuda_info["compute_capabilities"] = capabilities
            get_arch_list = getattr(torch.cuda, "get_arch_list", None)
            raw_architectures = get_arch_list() if callable(get_arch_list) else ()
            supported_architectures = [
                str(value) for value in (raw_architectures or ())
            ]
            if supported_architectures:
                cuda_info["supported_architectures"] = supported_architectures
            usable = True
            if capabilities and supported_architectures:
                usable = all(
                    f"sm_{major}{minor}" in supported_architectures
                    for major, minor in capabilities
                )
            cuda_info["usable"] = usable
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
        "paddle",
        "qdrant_client",
        "opentelemetry",
        "yase._native",
    ),
    *,
    probe_providers: bool = False,
) -> RuntimeInfo:
    """Inspect installed capabilities without importing heavy packages.

    Set ``probe_providers=True`` only for an explicit runtime readiness probe.
    """
    available = {
        package: importlib.util.find_spec(package) is not None
        for package in optional_packages
    }
    distribution_names = {
        "cv2": ("opencv-python", "opencv-python-headless"),
        "opentelemetry": ("opentelemetry-api",),
        "paddle": ("paddlepaddle", "paddlepaddle-gpu"),
        "qdrant_client": ("qdrant-client",),
        "yase._native": ("yase",),
    }
    versions: dict[str, str] = {}
    for package, installed in available.items():
        if not installed:
            continue
        for distribution in distribution_names.get(package, (package,)):
            try:
                versions[package] = importlib_metadata.version(distribution)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
    return RuntimeInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        numpy=np.__version__,
        cpu_count=os.cpu_count(),
        optional_packages=available,
        provider_info=collect_provider_info() if probe_providers else {},
        optional_versions=versions,
    )


def health_check(
    extractor: Any | None = None,
    *,
    required_packages: tuple[str, ...] = (),
    probe_providers: bool = False,
    required_providers: tuple[str, ...] = (),
) -> HealthReport:
    """Return readiness without downloading weights or invoking inference."""
    checks = {"numpy": True}
    details: dict[str, str] = {}
    runtime = collect_runtime_info(
        probe_providers=probe_providers or bool(required_providers)
    )
    for package in required_packages:
        available = _package_is_available(package)
        checks[f"package:{package}"] = available
        if not available:
            details[f"package:{package}"] = "not installed"

    onnx_info = runtime.provider_info.get("onnxruntime", {})
    available_providers = set(onnx_info.get("available_providers", ()))
    for provider in required_providers:
        available = provider in available_providers
        checks[f"provider:{provider}"] = available
        if not available:
            details[f"provider:{provider}"] = (
                "not available in the installed ONNX Runtime build"
            )

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
