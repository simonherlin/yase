"""Stable JSON-friendly serialization for semantic results."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .core import SemanticResult

RESULT_SCHEMA_VERSION = "1.0"
_RESULT_FIELDS = {
    "depth",
    "segmentation",
    "detections",
    "tags",
    "embeddings",
    "timestamp",
    "metadata",
    "ocr",
    "caption",
    "scene",
    "events",
    "keypoints",
    "relations",
    "document",
    "depth_map",
}


def _json_value(value: Any, include_arrays: bool) -> Any:
    if isinstance(value, np.ndarray):
        if include_arrays:
            return value.tolist()
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if is_dataclass(value):
        return {
            key: _json_value(item, include_arrays)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {
            str(key): _json_value(item, include_arrays) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, include_arrays) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def result_to_dict(result: SemanticResult, include_arrays: bool = False) -> dict:
    """Convert a result to a JSON-friendly mapping.

    By default large tensors are represented by shape and dtype only. Set
    ``include_arrays=True`` for small results or explicit archival exports.
    """
    if not isinstance(result, SemanticResult):
        raise TypeError("result must be a SemanticResult")
    payload = _json_value(asdict(result), include_arrays)
    payload["schema_version"] = RESULT_SCHEMA_VERSION
    return payload


def result_from_dict(payload: Mapping[str, Any]) -> SemanticResult:
    """Reconstruct a semantic result from a JSON-compatible mapping.

    Legacy payloads without ``schema_version`` are accepted as version 1.0.
    Shape/dtype-only array summaries cannot be losslessly reconstructed and
    are rejected instead of silently producing incorrect tensors.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    version = payload.get("schema_version", RESULT_SCHEMA_VERSION)
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported result schema version: {version}")
    values = {key: payload[key] for key in _RESULT_FIELDS if key in payload}
    for field in ("depth", "segmentation", "embeddings"):
        value = values.get(field)
        if isinstance(value, Mapping) and {"dtype", "shape"}.issubset(value):
            raise ValueError(
                f"{field} contains a shape/dtype summary; serialize with "
                "include_arrays=True to reconstruct it"
            )
    return SemanticResult(**values)


def result_to_json(
    result: SemanticResult,
    *,
    include_arrays: bool = False,
    indent: int | None = None,
) -> str:
    """Serialize one result as JSON."""
    return json.dumps(
        result_to_dict(result, include_arrays=include_arrays),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


def write_jsonl(
    results: Iterable[SemanticResult],
    destination: str | Path | TextIO,
    *,
    include_arrays: bool = False,
) -> int:
    """Write ordered results to JSON Lines and return the number written."""
    close = False
    if isinstance(destination, (str, Path)):
        handle = open(destination, "w", encoding="utf-8")
        close = True
    else:
        handle = destination
    count = 0
    try:
        for result in results:
            handle.write(result_to_json(result, include_arrays=include_arrays) + "\n")
            count += 1
    finally:
        if close:
            handle.close()
    return count


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "result_from_dict",
    "result_to_dict",
    "result_to_json",
    "write_jsonl",
]
