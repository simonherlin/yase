"""Explicit registry for optional model backends."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class BackendSpec:
    """Description of a registered backend factory."""

    name: str
    factory: Callable[..., Any]
    capabilities: tuple[str, ...] = ()
    extra: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BackendRegistry:
    """Register and instantiate backends without importing them eagerly."""

    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        capabilities: tuple[str, ...] = (),
        extra: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        replace: bool = False,
    ) -> BackendSpec:
        if not name or not callable(factory):
            raise TypeError("name must be non-empty and factory must be callable")
        if name in self._specs and not replace:
            raise ValueError(f"backend '{name}' is already registered")
        spec = BackendSpec(
            name=name,
            factory=factory,
            capabilities=tuple(capabilities),
            extra=extra,
            metadata=dict(metadata or {}),
        )
        self._specs[name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    def get(self, name: str) -> BackendSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs)) or "none"
            raise KeyError(f"unknown backend '{name}'; available: {available}") from exc

    def create(self, name: str, **options: Any) -> Any:
        return self.get(name).factory(**options)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[BackendSpec, ...]:
        return tuple(self._specs[name] for name in self.names())


def default_registry() -> BackendRegistry:
    """Return a registry containing only built-in, lazy-imported adapters."""
    from .adapters import (
        RFDETRExtractor,
        TransformersGroundingDinoExtractor,
        TransformersSAM3Extractor,
        TransformersSAM3VideoExtractor,
    )
    from .backends import OnnxRuntimeExtractor, TorchScriptExtractor
    from .runtimes import OpenVINOExtractor, TensorRTExtractor

    registry = BackendRegistry()
    registry.register(
        "torchscript",
        TorchScriptExtractor,
        capabilities=("depth", "segmentation", "batch"),
        extra="torch",
    )
    registry.register(
        "onnx",
        OnnxRuntimeExtractor,
        capabilities=("depth", "segmentation", "batch"),
        extra="onnx",
    )
    registry.register(
        "openvino",
        OpenVINOExtractor,
        capabilities=("depth", "segmentation", "batch", "cpu", "gpu", "npu"),
        extra="openvino",
    )
    registry.register(
        "tensorrt",
        TensorRTExtractor,
        capabilities=("depth", "segmentation", "batch", "cuda"),
        extra="tensorrt",
        metadata={"requires": "TensorRT plan and CUDA runtime"},
    )
    registry.register(
        "rf-detr",
        RFDETRExtractor,
        capabilities=("detection", "batch-compatible"),
        extra="rfdetr",
        metadata={
            "license": "Apache-2.0 for core/nano-large models; verify checkpoint"
        },
    )
    registry.register(
        "sam3",
        TransformersSAM3Extractor,
        capabilities=("segmentation", "open-vocabulary", "image"),
        extra="transformers",
        metadata={"license": "SAM License; inspect before redistribution"},
    )
    registry.register(
        "sam3-video",
        TransformersSAM3VideoExtractor,
        capabilities=("video", "segmentation", "tracking", "open-vocabulary"),
        extra="transformers",
        metadata={"license": "SAM License; inspect before redistribution"},
    )
    registry.register(
        "grounding-dino",
        TransformersGroundingDinoExtractor,
        capabilities=("detection", "open-vocabulary", "text-prompt"),
        extra="transformers",
        metadata={"license": "verify upstream checkpoint and code license"},
    )
    return registry


__all__ = ["BackendRegistry", "BackendSpec", "default_registry"]
