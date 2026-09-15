"""Measure Yase's optional native IoU/NMS kernels against Python fallbacks."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np

import yase.native as native


def _boxes(count: int) -> list[tuple[float, float, float, float]]:
    rng = np.random.default_rng(42)
    origins = rng.random((count, 2), dtype=np.float64) * 640.0
    sizes = rng.random((count, 2), dtype=np.float64) * 160.0 + 1.0
    return [
        (float(x), float(y), float(x + width), float(y + height))
        for (x, y), (width, height) in zip(origins, sizes)
    ]


def _measure(function: Callable[[], object], iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - started) / iterations


def _with_implementation(
    use_native: bool,
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iterations: int,
) -> dict[str, float]:
    saved_iou = native._native_iou_matrix
    saved_nms = native._native_nms_indices
    if not use_native:
        native._native_iou_matrix = None
        native._native_nms_indices = None
    try:
        return {
            "iou_ms": 1000.0
            * _measure(lambda: native.iou_matrix(boxes, boxes), iterations),
            "nms_ms": 1000.0
            * _measure(lambda: native.nms_indices(boxes, scores), iterations),
        }
    finally:
        native._native_iou_matrix = saved_iou
        native._native_nms_indices = saved_nms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.boxes < 1 or args.iterations < 1:
        parser.error("--boxes and --iterations must be positive")

    boxes = _boxes(args.boxes)
    scores = np.linspace(1.0, 0.0, args.boxes, dtype=np.float64).tolist()
    payload = {
        "boxes": args.boxes,
        "iterations": args.iterations,
        "native_available": native.NATIVE_AVAILABLE,
        "python": _with_implementation(False, boxes, scores, args.iterations),
    }
    if native.NATIVE_AVAILABLE:
        payload["native"] = _with_implementation(
            True, boxes, scores, args.iterations
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
