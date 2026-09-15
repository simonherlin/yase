"""Typed semantic primitives shared by model adapters.

The package deliberately keeps these structures small and model-agnostic. A
detector can expose rich data without forcing the core package to depend on a
particular framework or annotation format.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box in absolute ``(x1, y1, x2, y2)`` coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("x2/y2 must be greater than or equal to x1/y1")

    @classmethod
    def from_xywh(
        cls, x: float, y: float, width: float, height: float
    ) -> "BoundingBox":
        if width < 0 or height < 0:
            raise ValueError("width and height must be non-negative")
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError("a bounding box requires four values")
        return cls(*[float(value) for value in values])

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def clip(self, width: int, height: int) -> "BoundingBox":
        if width < 0 or height < 0:
            raise ValueError("image dimensions must be non-negative")
        return BoundingBox(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )


@dataclass(frozen=True)
class OrientedBoundingBox:
    """Rotated box in center/size coordinates with angle in radians."""

    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.width, self.height, self.angle)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("oriented-box values must be finite")
        if self.width < 0 or self.height < 0:
            raise ValueError("oriented-box width and height must be non-negative")

    def as_cxcywh_angle(self) -> tuple[float, float, float, float, float]:
        return self.center_x, self.center_y, self.width, self.height, self.angle


@dataclass(frozen=True)
class Detection:
    """One detected or grounded object."""

    label: str
    score: float
    box: BoundingBox
    mask: np.ndarray | None = None
    track_id: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("detection label must not be empty")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("detection score must be in [0, 1]")
        if not isinstance(self.box, BoundingBox):
            object.__setattr__(self, "box", BoundingBox.from_sequence(self.box))
        if self.mask is not None and not isinstance(self.mask, np.ndarray):
            object.__setattr__(self, "mask", np.asarray(self.mask))


@dataclass(frozen=True)
class Keypoint:
    """One named pose keypoint in image coordinates."""

    name: str
    x: float
    y: float
    score: float = 1.0
    visible: bool | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("keypoint name must not be empty")
        if not all(math.isfinite(float(value)) for value in (self.x, self.y)):
            raise ValueError("keypoint coordinates must be finite")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("keypoint score must be in [0, 1]")


@dataclass(frozen=True)
class Pose:
    """A named collection of keypoints for one person/object instance."""

    keypoints: tuple[Keypoint, ...]
    score: float = 1.0
    track_id: int | None = None
    skeleton: str | None = None

    def __post_init__(self) -> None:
        if not self.keypoints:
            raise ValueError("pose must contain at least one keypoint")
        if any(not isinstance(item, Keypoint) for item in self.keypoints):
            raise TypeError("pose keypoints must be Keypoint values")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("pose score must be in [0, 1]")


@dataclass(frozen=True)
class DepthMap:
    """Depth values with explicit metric/relative semantics and camera data."""

    values: np.ndarray
    unit: str = "relative"
    scale: float = 1.0
    invalid_value: float | None = None
    camera: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError("depth map values must be a 2-D array")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("depth map values must be numeric")
        if self.unit not in ("relative", "m", "cm", "mm"):
            raise ValueError("depth unit must be relative, m, cm, or mm")
        if not math.isfinite(float(self.scale)) or self.scale <= 0:
            raise ValueError("depth scale must be finite and positive")
        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(self, "camera", dict(self.camera))


@dataclass(frozen=True)
class Relation:
    """A scored subject-predicate-object relation from a scene graph."""

    subject: Any
    predicate: str
    object: Any
    score: float = 1.0
    evidence: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.predicate:
            raise ValueError("relation predicate must not be empty")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("relation score must be in [0, 1]")
        if any(not item for item in self.evidence):
            raise ValueError("relation evidence names must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TextRegion:
    """OCR result with optional geometry and language metadata."""

    text: str
    score: float = 1.0
    box: BoundingBox | None = None
    polygon: np.ndarray | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("OCR score must be in [0, 1]")
        if self.box is not None and not isinstance(self.box, BoundingBox):
            object.__setattr__(self, "box", BoundingBox.from_sequence(self.box))
        if self.polygon is not None and not isinstance(self.polygon, np.ndarray):
            object.__setattr__(self, "polygon", np.asarray(self.polygon))


@dataclass(frozen=True)
class SemanticEvent:
    """A temporal event emitted by a video or streaming backend."""

    label: str
    score: float
    start: float
    end: float | None = None
    track_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("event label must not be empty")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("event score must be in [0, 1]")
        if self.end is not None and self.end < self.start:
            raise ValueError("event end must be greater than or equal to start")


__all__ = [
    "BoundingBox",
    "DepthMap",
    "Detection",
    "Keypoint",
    "OrientedBoundingBox",
    "Pose",
    "Relation",
    "SemanticEvent",
    "TextRegion",
]
