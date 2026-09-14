"""Framework-neutral normalization for common detector outputs.

Detection models disagree on whether boxes are ``xyxy`` or ``xywh`` and on
whether outputs are dictionaries, columnar arrays, or rows. This module keeps
that compatibility logic outside the core schema and does not import any
modeling framework.
"""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from .schema import BoundingBox, Detection

_STRUCTURAL_KEYS = {
    "box",
    "bbox",
    "boxes",
    "xyxy",
    "xywh",
    "score",
    "scores",
    "confidence",
    "confidences",
    "label",
    "labels",
    "class",
    "class_id",
    "class_ids",
    "class_name",
    "names",
    "name",
    "mask",
    "masks",
    "track_id",
    "track_ids",
}


def _label(value: Any, label_map: Optional[Mapping[Any, str]]) -> str:
    if value is None:
        return "object"
    if label_map is not None and value in label_map:
        return str(label_map[value])
    if isinstance(value, (int, np.integer)):
        return f"class_{int(value)}"
    return str(value)


def _score(value: Any, strict: bool) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("detection score must be finite")
    if strict and not 0 <= result <= 1:
        raise ValueError("detection score must be in [0, 1]")
    return min(1.0, max(0.0, result))


def _box(
    values: Any,
    box_format: str,
    image_shape: Optional[tuple[int, int]],
    clip: bool,
) -> BoundingBox:
    coordinates = [float(value) for value in np.asarray(values).reshape(-1)]
    if len(coordinates) != 4:
        raise ValueError("a detection box requires four values")
    normalized = box_format.startswith("normalized_")
    format_name = box_format.removeprefix("normalized_")
    if format_name not in {"xyxy", "xywh"}:
        raise ValueError(
            "box_format must be xyxy, xywh, normalized_xyxy, or normalized_xywh"
        )
    if normalized:
        if image_shape is None:
            raise ValueError("image_shape is required for normalized boxes")
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape must contain positive height and width")
        if format_name == "xyxy":
            coordinates[0] *= width
            coordinates[2] *= width
            coordinates[1] *= height
            coordinates[3] *= height
        else:
            coordinates[0] *= width
            coordinates[2] *= width
            coordinates[1] *= height
            coordinates[3] *= height
    if format_name == "xywh":
        result = BoundingBox.from_xywh(*coordinates)
    else:
        result = BoundingBox.from_sequence(coordinates)
    return (
        result.clip(image_shape[1], image_shape[0]) if clip and image_shape else result
    )


def _single_mapping(
    item: Mapping[str, Any],
    *,
    image_shape: Optional[tuple[int, int]],
    box_format: str,
    label_map: Optional[Mapping[Any, str]],
    score_threshold: float,
    clip: bool,
    strict: bool,
) -> Optional[Detection]:
    values = item.get("box", item.get("bbox", item.get("xyxy", item.get("xywh"))))
    if values is None and all(key in item for key in ("x1", "y1", "x2", "y2")):
        values = [item[key] for key in ("x1", "y1", "x2", "y2")]
    if values is None:
        raise ValueError("detection mapping must contain box/bbox/xyxy/xywh")
    local_format = box_format
    if "xywh" in item and "box" not in item and "bbox" not in item:
        local_format = "normalized_xywh" if box_format == "normalized_xyxy" else "xywh"
    score = _score(item.get("score", item.get("confidence", 1.0)), strict)
    if score < score_threshold:
        return None
    class_value = item.get("label", item.get("class_name", item.get("name")))
    if class_value is None:
        class_value = item.get("class", item.get("class_id"))
    attributes = {
        key: value for key, value in item.items() if key not in _STRUCTURAL_KEYS
    }
    if "class_id" in item:
        attributes.setdefault("class_id", item["class_id"])
    track_id = item.get("track_id")
    return Detection(
        label=_label(class_value, label_map),
        score=score,
        box=_box(values, local_format, image_shape, clip),
        mask=item.get("mask"),
        track_id=int(track_id) if track_id is not None else None,
        attributes=attributes,
    )


def _columnar_mapping(
    value: Mapping[str, Any],
    *,
    image_shape: Optional[tuple[int, int]],
    box_format: str,
    label_map: Optional[Mapping[Any, str]],
    score_threshold: float,
    clip: bool,
    strict: bool,
) -> list[Detection]:
    boxes = value.get("boxes", value.get("bboxes"))
    if boxes is None:
        return []
    boxes_list = list(np.asarray(boxes))

    def columns(item: Any, default: Any) -> list[Any]:
        if item is None:
            return [default] * len(boxes_list)
        array = np.asarray(item)
        if array.ndim == 0:
            return [array.item()] * len(boxes_list)
        return list(array)

    scores = columns(value.get("scores", value.get("confidences")), 1.0)
    labels = columns(value.get("labels", value.get("class_ids")), None)
    masks = columns(value.get("masks"), None)
    track_ids = columns(value.get("track_ids"), None)
    detections = []
    for index, box in enumerate(boxes_list):
        item = {
            "box": box,
            "score": scores[index],
            "label": labels[index],
            "mask": masks[index],
            "track_id": track_ids[index],
        }
        detection = _single_mapping(
            item,
            image_shape=image_shape,
            box_format=box_format,
            label_map=label_map,
            score_threshold=score_threshold,
            clip=clip,
            strict=strict,
        )
        if detection is not None:
            detections.append(detection)
    return detections


def normalise_detections(
    value: Any,
    *,
    image_shape: Optional[tuple[int, int]] = None,
    box_format: str = "xyxy",
    label_map: Optional[Mapping[Any, str]] = None,
    score_threshold: float = 0.0,
    clip: bool = False,
    strict: bool = False,
) -> list[Detection]:
    """Convert common detector output shapes into typed ``Detection`` values.

    Supported forms include typed detections, single mappings, columnar
    mappings (``boxes``/``scores``/``labels``), mapping lists, and numeric rows
    shaped ``(N, 6)`` as ``x1,y1,x2,y2,score,class_id``. Scores are clipped to
    ``[0, 1]`` by default; ``strict=True`` rejects out-of-range scores.
    """
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1]")
    if isinstance(value, Detection):
        return [value] if value.score >= score_threshold else []
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and not value:
        return []
    if isinstance(value, Mapping):
        nested = value.get("detections", value.get("predictions", value.get("objects")))
        if nested is not None:
            return normalise_detections(
                nested,
                image_shape=image_shape,
                box_format=box_format,
                label_map=label_map,
                score_threshold=score_threshold,
                clip=clip,
                strict=strict,
            )
        if "boxes" in value or "bboxes" in value:
            return _columnar_mapping(
                value,
                image_shape=image_shape,
                box_format=box_format,
                label_map=label_map,
                score_threshold=score_threshold,
                clip=clip,
                strict=strict,
            )
        detection = _single_mapping(
            value,
            image_shape=image_shape,
            box_format=box_format,
            label_map=label_map,
            score_threshold=score_threshold,
            clip=clip,
            strict=strict,
        )
        return [detection] if detection is not None else []
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (Detection, Mapping)) for item in value):
            detections: list[Detection] = []
            for item in value:
                detections.extend(
                    normalise_detections(
                        item,
                        image_shape=image_shape,
                        box_format=box_format,
                        label_map=label_map,
                        score_threshold=score_threshold,
                        clip=clip,
                        strict=strict,
                    )
                )
            return detections
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError("array detections must have shape (N, >=5)")
    if array.shape[1] == 5:
        array = np.column_stack((array, np.zeros(len(array))))
    return normalise_detections(
        [
            {
                "box": row[:4],
                "score": row[4],
                "class_id": int(row[5]),
            }
            for row in array
        ],
        image_shape=image_shape,
        box_format=box_format,
        label_map=label_map,
        score_threshold=score_threshold,
        clip=clip,
        strict=strict,
    )


__all__ = ["normalise_detections"]
