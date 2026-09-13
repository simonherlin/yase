"""Public, dependency-light API for semantic extraction.

Heavy model backends are loaded lazily so importing yase never downloads
weights or requires PyTorch.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np

ImageInput = Union[str, Path, np.ndarray, Any]


@dataclass(frozen=True)
class SemanticResult:
    """The semantic outputs produced for one image."""

    depth: Optional[np.ndarray] = None
    segmentation: Optional[np.ndarray] = None
    detections: Optional[Any] = None
    tags: Optional[Any] = None
    embeddings: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("depth", "segmentation", "embeddings"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, np.ndarray):
                object.__setattr__(self, name, np.asarray(value))


class Extractor(Protocol):
    """Minimal protocol implemented by model backends."""

    def extract(self, image: ImageInput) -> SemanticResult: ...


def load_image(image: ImageInput, color_order: str = "RGB") -> np.ndarray:
    """Return an HxWxC uint8 RGB array.

    Paths and Pillow images are converted to RGB. Numpy arrays are copied only
    when needed; arrays are assumed RGB by default (use BGR explicitly for
    OpenCV frames).
    """
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load image paths") from exc
        with Image.open(image) as pil_image:
            return np.asarray(pil_image.convert("RGB"))
    if hasattr(image, "convert") and hasattr(image, "size"):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must have 2 or 3 dimensions")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] not in (3, 4):
        raise ValueError("image must have 3 channels (or 4 with alpha)")
    array = array[..., :3]
    if color_order.upper() == "BGR":
        array = array[..., ::-1]
    elif color_order.upper() != "RGB":
        raise ValueError("color_order must be RGB or BGR")
    return np.ascontiguousarray(array)


def _normalise_output(
    output: Any, task: str, timestamp: Optional[float]
) -> SemanticResult:
    """Adapt common backend return values to SemanticResult."""
    if isinstance(output, SemanticResult):
        if timestamp is None:
            return output
        return SemanticResult(
            depth=output.depth,
            segmentation=output.segmentation,
            detections=output.detections,
            tags=output.tags,
            embeddings=output.embeddings,
            timestamp=timestamp,
            metadata=output.metadata,
        )
    if isinstance(output, Mapping):
        return SemanticResult(
            depth=output.get("depth"),
            segmentation=output.get("segmentation", output.get("mask")),
            detections=output.get("detections"),
            tags=output.get("tags"),
            embeddings=output.get("embeddings"),
            timestamp=timestamp,
            metadata={
                k: v
                for k, v in output.items()
                if k
                not in (
                    "depth",
                    "segmentation",
                    "mask",
                    "detections",
                    "tags",
                    "embeddings",
                )
            },
        )
    if task == "depth":
        return SemanticResult(depth=np.asarray(output), timestamp=timestamp)
    if task == "segmentation":
        return SemanticResult(segmentation=np.asarray(output), timestamp=timestamp)
    if task == "both" and isinstance(output, (tuple, list)) and len(output) == 2:
        return SemanticResult(
            depth=np.asarray(output[0]),
            segmentation=np.asarray(output[1]),
            timestamp=timestamp,
        )
    raise TypeError("backend output must be SemanticResult, mapping, or array")


class Yase:
    """Semantic extractor with an injectable backend.

    extractor is useful for applications and tests. Without one, the
    Monodepth2 backend is imported and initialized on the first extraction.
    """

    def __init__(
        self,
        task: str = "depth",
        extractor: Optional[Any] = None,
        model: str = "monodepth2",
        color_order: str = "RGB",
        **backend_options: Any,
    ) -> None:
        if task not in ("depth", "segmentation", "both"):
            raise ValueError("task must be 'depth', 'segmentation', or 'both'")
        self.task = task
        self.model = model
        self.color_order = color_order
        self._extractor = extractor
        self._backend_options = backend_options

    @property
    def extractor(self) -> Any:
        if self._extractor is None:
            if self.model == "torchscript":
                from .backends import TorchScriptExtractor

                self._extractor = TorchScriptExtractor(**self._backend_options)
            else:
                raise ValueError(
                    "no backend is loaded by default; pass extractor=... or "
                    'use model="torchscript" with model_path=...'
                )
        return self._extractor

    def extract(
        self, image: ImageInput, timestamp: Optional[float] = None
    ) -> SemanticResult:
        """Extract semantics from one image."""
        rgb = load_image(image, color_order=self.color_order)
        backend = self.extractor
        if hasattr(backend, "extract"):
            output = backend.extract(rgb)
        elif hasattr(backend, "predict"):
            output = backend.predict(rgb)
        elif self.task == "depth" and hasattr(backend, "predict_depth"):
            output = backend.predict_depth(rgb)
        elif callable(backend):
            output = backend(rgb)
        else:
            raise TypeError("extractor must be callable or expose extract/predict")
        return _normalise_output(output, self.task, timestamp)

    def run_inference(
        self, input_data: ImageInput, timestamp: Optional[float] = None
    ) -> SemanticResult:
        """Backward-compatible alias for extract."""
        return self.extract(input_data, timestamp=timestamp)

    def __call__(
        self, image: ImageInput, timestamp: Optional[float] = None
    ) -> SemanticResult:
        return self.extract(image, timestamp=timestamp)


__all__ = ["Extractor", "ImageInput", "SemanticResult", "Yase", "load_image"]
