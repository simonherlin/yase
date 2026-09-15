"""Explicit registry for optional model backends."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
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

    def discover_entry_points(
        self,
        group: str = "yase.backends",
        *,
        replace: bool = False,
    ) -> tuple[BackendSpec, ...]:
        """Load third-party backend factories from Python entry points.

        Discovery is explicit so importing Yase never executes arbitrary plugin
        code. Both the modern ``select`` API and Python 3.9's mapping-shaped
        entry-point API are supported.
        """
        if not group:
            raise ValueError("entry-point group must not be empty")
        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            entries = discovered.select(group=group)
        else:  # pragma: no cover - Python 3.9 compatibility branch
            entries = discovered.get(group, ())
        loaded: list[BackendSpec] = []
        for entry in entries:
            try:
                factory = entry.load()
                spec = self.register(
                    entry.name,
                    factory,
                    metadata={"entry_point": entry.value, "group": group},
                    replace=replace,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"could not load backend plugin '{entry.name}' from {group}"
                ) from exc
            loaded.append(spec)
        return tuple(loaded)


def default_registry(*, include_plugins: bool = False) -> BackendRegistry:
    """Return built-ins, optionally extended by explicit plugin discovery."""
    from .adapters import (
        PaddleOCRExtractor,
        RFDETRExtractor,
        TesseractExtractor,
        TransformersGroundingDinoExtractor,
        TransformersImageEmbeddingExtractor,
        TransformersSAM3Extractor,
        TransformersSAM3VideoExtractor,
        TransformersVLMExtractor,
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
    registry.register(
        "image-embedding",
        TransformersImageEmbeddingExtractor,
        capabilities=("embedding", "retrieval", "batch", "image"),
        extra="transformers",
    )
    registry.register(
        "vlm",
        TransformersVLMExtractor,
        capabilities=("caption", "question-answering", "structured-output", "image"),
        extra="transformers",
    )
    registry.register(
        "tesseract",
        TesseractExtractor,
        capabilities=("ocr", "text-detection", "image"),
        extra="ocr",
    )
    registry.register(
        "paddleocr",
        PaddleOCRExtractor,
        capabilities=("ocr", "text-detection", "image"),
        extra="paddle",
    )
    if include_plugins:
        registry.discover_entry_points()
    return registry


__all__ = ["BackendRegistry", "BackendSpec", "default_registry"]
