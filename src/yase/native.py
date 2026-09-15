"""Optional native acceleration with a deterministic Python fallback."""

import math
from collections.abc import Sequence
from typing import Any, Optional

try:  # The binary is built explicitly; importing Yase never compiles code.
    from ._native import iou_matrix as _native_iou_matrix
    from ._native import nms_indices as _native_nms_indices
except ImportError:  # pragma: no cover - exercised when no compiler artifact exists
    _native_iou_matrix = None
    _native_nms_indices = None


NATIVE_AVAILABLE = _native_iou_matrix is not None


def _box_values(box: Any) -> tuple[float, float, float, float]:
    values = box.as_xyxy() if hasattr(box, "as_xyxy") else box
    if len(values) < 4:
        raise ValueError("box must contain at least four values")
    return tuple(float(value) for value in values[:4])  # type: ignore[return-value]


def _python_iou_matrix(
    left: Sequence[tuple[float, float, float, float]],
    right: Sequence[tuple[float, float, float, float]],
) -> list[list[float]]:
    return [
        [_python_iou(left_box, right_box) for right_box in right] for left_box in left
    ]


def _python_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def iou_matrix(left: Sequence[Any], right: Sequence[Any]) -> list[list[float]]:
    """Return pairwise IoU values, using C++ when the optional module exists."""
    normalized_left = [_box_values(box) for box in left]
    normalized_right = [_box_values(box) for box in right]
    if _native_iou_matrix is not None:
        return _native_iou_matrix(normalized_left, normalized_right)
    return _python_iou_matrix(normalized_left, normalized_right)


def _python_nms_indices(
    boxes: Sequence[tuple[float, float, float, float]],
    scores: Sequence[float],
    iou_threshold: float,
    class_ids: Optional[Sequence[int]],
) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
    kept: list[int] = []
    for index in order:
        if all(
            class_ids is None
            or class_ids[index] != class_ids[other]
            or _python_iou(boxes[index], boxes[other]) <= iou_threshold
            for other in kept
        ):
            kept.append(index)
    return kept


def nms_indices(
    boxes: Sequence[Any],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
    class_ids: Optional[Sequence[int]] = None,
) -> list[int]:
    """Return stable greedy-NMS indices, using C++ when available."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    normalized_boxes = [_box_values(box) for box in boxes]
    normalized_scores = [float(score) for score in scores]
    if len(normalized_boxes) != len(normalized_scores):
        raise ValueError("boxes and scores must have the same length")
    if not all(math.isfinite(score) for score in normalized_scores):
        raise ValueError("scores must be finite")
    normalized_classes = (
        None if class_ids is None else [int(value) for value in class_ids]
    )
    if normalized_classes is not None and len(normalized_classes) != len(
        normalized_boxes
    ):
        raise ValueError("boxes and class_ids must have the same length")
    if _native_nms_indices is not None:
        return _native_nms_indices(
            normalized_boxes, normalized_scores, iou_threshold, normalized_classes
        )
    return _python_nms_indices(
        normalized_boxes, normalized_scores, iou_threshold, normalized_classes
    )


__all__ = ["NATIVE_AVAILABLE", "iou_matrix", "nms_indices"]
