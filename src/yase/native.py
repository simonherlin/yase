"""Optional native acceleration with a deterministic Python fallback."""

from collections.abc import Sequence
from typing import Any

try:  # The binary is built explicitly; importing Yase never compiles code.
    from ._native import iou_matrix as _native_iou_matrix
except ImportError:  # pragma: no cover - exercised when no compiler artifact exists
    _native_iou_matrix = None


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
    result = []
    for left_box in left:
        row = []
        for right_box in right:
            x1 = max(left_box[0], right_box[0])
            y1 = max(left_box[1], right_box[1])
            x2 = min(left_box[2], right_box[2])
            y2 = min(left_box[3], right_box[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            left_area = max(0.0, left_box[2] - left_box[0]) * max(
                0.0, left_box[3] - left_box[1]
            )
            right_area = max(0.0, right_box[2] - right_box[0]) * max(
                0.0, right_box[3] - right_box[1]
            )
            union = left_area + right_area - intersection
            row.append(intersection / union if union else 0.0)
        result.append(row)
    return result


def iou_matrix(left: Sequence[Any], right: Sequence[Any]) -> list[list[float]]:
    """Return pairwise IoU values, using C++ when the optional module exists."""
    normalized_left = [_box_values(box) for box in left]
    normalized_right = [_box_values(box) for box in right]
    if _native_iou_matrix is not None:
        return _native_iou_matrix(normalized_left, normalized_right)
    return _python_iou_matrix(normalized_left, normalized_right)


__all__ = ["NATIVE_AVAILABLE", "iou_matrix"]
