"""Versioned, model-neutral observation contracts.

The legacy :class:`~yase.core.SemanticResult` remains the lightweight return
value for one-shot extraction.  ``ObservationBundle`` is the durable contract
for pipelines, video streams, archives, and downstream systems: it binds a
result to a frame, records which models produced it, and keeps uncertainty
explicit instead of hiding it in backend-specific metadata.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .core import SemanticResult
from .serialization import _json_value, result_from_dict, result_to_dict


def _copy_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class FrameRef:
    """Stable identity and acquisition metadata for one frame."""

    frame_id: int
    source_id: str = "default"
    timestamp: float | None = None
    width: int | None = None
    height: int | None = None
    color_order: str = "RGB"

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if self.timestamp is not None and not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite when provided")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be positive when provided")
        order = self.color_order.upper()
        if order not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")
        object.__setattr__(self, "color_order", order)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "color_order": self.color_order,
        }


@dataclass(frozen=True)
class ModelProvenance:
    """Reproducibility and redistribution metadata for one model artifact."""

    model_id: str
    revision: str | None = None
    artifact_sha256: str | None = None
    runtime: str | None = None
    device: str | None = None
    precision: str | None = None
    license: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.artifact_sha256 is not None:
            digest = self.artifact_sha256.lower()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("artifact_sha256 must be a 64-character hex digest")
            object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "runtime": self.runtime,
            "device": self.device,
            "precision": self.precision,
            "license": self.license,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Uncertainty:
    """Confidence semantics attached to a named output or stage."""

    confidence: float | None = None
    calibrated: bool = False
    method: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None:
            if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "calibrated": self.calibrated,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingRecord:
    """An embedding with its vector space and normalization semantics."""

    vector: np.ndarray
    space: str
    model_id: str
    revision: str | None = None
    modality: str = "image"
    normalized: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding vector must be a non-empty 1-D array")
        if not np.issubdtype(vector.dtype, np.number) or not np.all(
            np.isfinite(vector)
        ):
            raise ValueError("embedding vector must contain finite numbers")
        if not self.space or not self.model_id or not self.modality:
            raise ValueError("space, model_id, and modality must not be empty")
        if self.normalized and not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4):
            raise ValueError("normalized embedding vector must have unit norm")
        object.__setattr__(self, "vector", np.ascontiguousarray(vector))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self, include_values: bool = False) -> dict:
        payload = {
            "space": self.space,
            "model_id": self.model_id,
            "revision": self.revision,
            "modality": self.modality,
            "normalized": self.normalized,
            "shape": list(self.vector.shape),
            "dtype": str(self.vector.dtype),
            "metadata": dict(self.metadata),
        }
        if include_values:
            payload["vector"] = self.vector.tolist()
        return payload


@dataclass(frozen=True)
class ObservationBundle:
    """A versioned observation suitable for storage or stream transport."""

    frame: FrameRef
    result: SemanticResult
    schema_version: str = "1.0"
    provenance: tuple[ModelProvenance, ...] = ()
    uncertainty: Mapping[str, Uncertainty] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_version or "." not in self.schema_version:
            raise ValueError("schema_version must be a non-empty major.minor value")
        if not isinstance(self.frame, FrameRef):
            raise TypeError("frame must be a FrameRef")
        if not isinstance(self.result, SemanticResult):
            raise TypeError("result must be a SemanticResult")
        if any(not isinstance(item, ModelProvenance) for item in self.provenance):
            raise TypeError("provenance must contain ModelProvenance values")
        if any(not isinstance(item, Uncertainty) for item in self.uncertainty.values()):
            raise TypeError("uncertainty values must be Uncertainty instances")
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "uncertainty", dict(self.uncertainty))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @classmethod
    def from_result(
        cls,
        result: SemanticResult,
        frame: FrameRef | None = None,
        *,
        frame_id: int = 0,
        source_id: str = "default",
        width: int | None = None,
        height: int | None = None,
        color_order: str = "RGB",
        provenance: tuple[ModelProvenance, ...] = (),
        uncertainty: Mapping[str, Uncertainty] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ObservationBundle":
        """Wrap a legacy result without changing its contents."""
        if frame is None:
            frame = FrameRef(
                frame_id=frame_id,
                source_id=source_id,
                timestamp=result.timestamp,
                width=width,
                height=height,
                color_order=color_order,
            )
        return cls(
            frame=frame,
            result=result,
            provenance=provenance,
            uncertainty=uncertainty or {},
            metadata=metadata or {},
        )

    def to_dict(self, include_arrays: bool = False) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "frame": self.frame.to_dict(),
            "result": result_to_dict(self.result, include_arrays=include_arrays),
            "provenance": [item.to_dict() for item in self.provenance],
            "uncertainty": {
                key: value.to_dict() for key, value in self.uncertainty.items()
            },
            "metadata": dict(self.metadata),
        }
        return _json_value(payload, include_arrays)


def observation_to_json(
    observation: ObservationBundle,
    *,
    include_arrays: bool = False,
    indent: int | None = None,
) -> str:
    """Serialize one observation bundle to deterministic JSON."""
    if not isinstance(observation, ObservationBundle):
        raise TypeError("observation must be an ObservationBundle")
    return json.dumps(
        observation.to_dict(include_arrays=include_arrays),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


def observation_from_dict(payload: Mapping[str, Any]) -> ObservationBundle:
    """Reconstruct an observation bundle from a JSON-compatible mapping.

    Array summaries remain intentionally non-reconstructable. Callers that
    need tensor values must provide a payload produced with
    ``include_arrays=True`` or restore the arrays from an external archive.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    version = payload.get("schema_version", "1.0")
    if version != "1.0":
        raise ValueError(f"unsupported observation schema version: {version}")
    frame_payload = payload.get("frame")
    result_payload = payload.get("result")
    if not isinstance(frame_payload, Mapping):
        raise TypeError("observation frame must be a mapping")
    if not isinstance(result_payload, Mapping):
        raise TypeError("observation result must be a mapping")
    provenance_payload = payload.get("provenance", [])
    if not isinstance(provenance_payload, (list, tuple)):
        raise TypeError("observation provenance must be a list")
    uncertainty_payload = payload.get("uncertainty", {})
    if not isinstance(uncertainty_payload, Mapping):
        raise TypeError("observation uncertainty must be a mapping")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("observation metadata must be a mapping")
    provenance = []
    for item in provenance_payload:
        if not isinstance(item, Mapping):
            raise TypeError("provenance entries must be mappings")
        provenance.append(ModelProvenance(**dict(item)))
    uncertainty = {}
    for key, value in uncertainty_payload.items():
        if not isinstance(value, Mapping):
            raise TypeError("uncertainty entries must be mappings")
        uncertainty[str(key)] = Uncertainty(**dict(value))
    return ObservationBundle(
        frame=FrameRef(**dict(frame_payload)),
        result=result_from_dict(result_payload),
        schema_version=str(version),
        provenance=tuple(provenance),
        uncertainty=uncertainty,
        metadata=dict(metadata),
    )


def write_observation_jsonl(
    observations: Any,
    destination: str | Path | TextIO,
    *,
    include_arrays: bool = False,
) -> int:
    """Write ordered observation bundles to JSON Lines and return the count."""
    close = False
    if isinstance(destination, (str, Path)):
        handle = open(destination, "w", encoding="utf-8")
        close = True
    else:
        handle = destination
    count = 0
    try:
        for observation in observations:
            handle.write(
                observation_to_json(observation, include_arrays=include_arrays) + "\n"
            )
            count += 1
    finally:
        if close:
            handle.close()
    return count


__all__ = [
    "EmbeddingRecord",
    "FrameRef",
    "ModelProvenance",
    "ObservationBundle",
    "Uncertainty",
    "observation_from_dict",
    "observation_to_json",
    "write_observation_jsonl",
]
