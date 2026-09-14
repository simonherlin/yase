"""Small deterministic quality metrics for semantic extraction benchmarks."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .schema import Detection
from .tracking import box_iou


@dataclass(frozen=True)
class DetectionMetrics:
    """Aggregate detection quality at one IoU threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    mean_iou: float
    iou_threshold: float

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Union[float, int]]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "mean_iou": self.mean_iou,
            "iou_threshold": self.iou_threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class AveragePrecisionResult:
    """Interpolated precision-recall summary for one IoU threshold."""

    average_precision: float
    average_recall: float
    iou_threshold: float
    ground_truth_count: int
    prediction_count: int
    true_positives: int
    false_positives: int
    precision: tuple[float, ...] = ()
    recall: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "average_precision": self.average_precision,
            "average_recall": self.average_recall,
            "iou_threshold": self.iou_threshold,
            "ground_truth_count": self.ground_truth_count,
            "prediction_count": self.prediction_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "precision": list(self.precision),
            "recall": list(self.recall),
        }


@dataclass(frozen=True)
class MeanAveragePrecisionResult:
    """Average precision and recall over a sequence of IoU thresholds."""

    iou_thresholds: tuple[float, ...]
    per_threshold: tuple[AveragePrecisionResult, ...]
    mean_average_precision: float
    mean_average_recall: float

    def to_dict(self) -> dict[str, object]:
        return {
            "iou_thresholds": list(self.iou_thresholds),
            "per_threshold": [item.to_dict() for item in self.per_threshold],
            "mean_average_precision": self.mean_average_precision,
            "mean_average_recall": self.mean_average_recall,
        }


def evaluate_detections(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> DetectionMetrics:
    """Compute greedy one-to-one precision/recall and mean matched IoU."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    predicted = sorted(list(predictions), key=lambda item: item.score, reverse=True)
    expected = list(ground_truth)
    used: set[int] = set()
    overlaps: list[float] = []
    true_positives = 0
    for detection in predicted:
        best_index: Optional[int] = None
        best_overlap = iou_threshold
        for index, target in enumerate(expected):
            if index in used or (class_aware and detection.label != target.label):
                continue
            overlap = box_iou(detection.box, target.box)
            if overlap >= best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None:
            used.add(best_index)
            true_positives += 1
            overlaps.append(best_overlap)
    false_positives = len(predicted) - true_positives
    false_negatives = len(expected) - true_positives
    return DetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        mean_iou=float(np.mean(overlaps)) if overlaps else 0.0,
        iou_threshold=iou_threshold,
    )


def _as_detection_frames(
    values: Iterable[Iterable[Detection]],
) -> list[list[Detection]]:
    """Materialize either one frame of detections or aligned detection frames."""
    materialized = list(values)
    if not materialized:
        return []
    if all(isinstance(item, Detection) for item in materialized):
        return [materialized]
    return [list(frame) for frame in materialized]


def _average_precision_at_threshold(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_threshold: float,
    class_aware: bool,
    recall_thresholds: tuple[float, ...],
    overlap_function: Any = box_iou,
    require_masks: bool = False,
) -> AveragePrecisionResult:
    predicted_frames = _as_detection_frames(predictions)
    truth_frames = _as_detection_frames(ground_truth)
    if len(predicted_frames) != len(truth_frames):
        raise ValueError("predictions and ground_truth must have equal frame counts")
    total_ground_truth = 0
    for _, expected in zip(predicted_frames, truth_frames):
        total_ground_truth += (
            sum(item.mask is not None for item in expected)
            if require_masks
            else len(expected)
        )
    # Sorting by score then frame/order makes equal-score outputs deterministic.
    indexed_candidates: list[tuple[float, int, int, Detection]] = []
    for frame_index, frame in enumerate(predicted_frames):
        for order, item in enumerate(frame):
            if require_masks and item.mask is None:
                continue
            indexed_candidates.append((float(item.score), frame_index, order, item))
    indexed_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used: dict[int, set[int]] = {index: set() for index in range(len(truth_frames))}
    true_positive_flags: list[int] = []
    false_positive_flags: list[int] = []
    for _, frame_index, _, prediction in indexed_candidates:
        expected = truth_frames[frame_index]
        best_index: Optional[int] = None
        best_overlap = iou_threshold
        for target_index, target in enumerate(expected):
            if target_index in used[frame_index]:
                continue
            if require_masks and target.mask is None:
                continue
            if class_aware and prediction.label != target.label:
                continue
            overlap = (
                overlap_function(prediction.mask, target.mask)
                if require_masks
                else overlap_function(prediction.box, target.box)
            )
            if overlap >= best_overlap:
                best_overlap = overlap
                best_index = target_index
        if best_index is None:
            true_positive_flags.append(0)
            false_positive_flags.append(1)
        else:
            used[frame_index].add(best_index)
            true_positive_flags.append(1)
            false_positive_flags.append(0)
    if total_ground_truth:
        true_positives = np.cumsum(true_positive_flags, dtype=np.int64)
        false_positives = np.cumsum(false_positive_flags, dtype=np.int64)
        recalls = true_positives / float(total_ground_truth)
        precisions = true_positives / np.maximum(true_positives + false_positives, 1)
        envelope = np.maximum.accumulate(precisions[::-1])[::-1]
        sampled = np.asarray(
            [
                float(envelope[np.flatnonzero(recalls >= threshold)[0]])
                if np.any(recalls >= threshold)
                else 0.0
                for threshold in recall_thresholds
            ],
            dtype=np.float64,
        )
        average_precision = float(np.mean(sampled))
        average_recall = float(recalls[-1]) if len(recalls) else 0.0
        precision_values = tuple(float(value) for value in precisions)
        recall_values = tuple(float(value) for value in recalls)
        tp_count = int(true_positives[-1]) if len(true_positives) else 0
        fp_count = int(false_positives[-1]) if len(false_positives) else 0
    else:
        average_precision = average_recall = 0.0
        precision_values = recall_values = ()
        tp_count = fp_count = 0
    return AveragePrecisionResult(
        average_precision=average_precision,
        average_recall=average_recall,
        iou_threshold=iou_threshold,
        ground_truth_count=total_ground_truth,
        prediction_count=len(indexed_candidates),
        true_positives=tp_count,
        false_positives=fp_count,
        precision=precision_values,
        recall=recall_values,
    )


def evaluate_average_precision(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
    recall_thresholds: Optional[Iterable[float]] = None,
) -> AveragePrecisionResult:
    """Compute COCO-style 101-point interpolated AP at one IoU threshold.

    Inputs are aligned frames/images. A flat iterable of detections is treated
    as one image. This lightweight evaluator intentionally omits COCO area,
    ignore-region, and max-detections dimensions.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    thresholds = (
        tuple(np.linspace(0.0, 1.0, 101))
        if recall_thresholds is None
        else tuple(float(value) for value in recall_thresholds)
    )
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("recall_thresholds must be non-empty values in [0, 1]")
    return _average_precision_at_threshold(
        predictions,
        ground_truth,
        iou_threshold,
        class_aware,
        thresholds,
    )


def evaluate_mean_average_precision(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_thresholds: Optional[Iterable[float]] = None,
    class_aware: bool = True,
) -> MeanAveragePrecisionResult:
    """Compute mean AP over the standard ``0.50:0.05:0.95`` IoU sweep."""
    thresholds = (
        tuple(round(float(value), 2) for value in np.arange(0.5, 0.96, 0.05))
        if iou_thresholds is None
        else tuple(float(value) for value in iou_thresholds)
    )
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("iou_thresholds must be non-empty values in [0, 1]")
    predicted_frames = _as_detection_frames(predictions)
    truth_frames = _as_detection_frames(ground_truth)
    results = tuple(
        evaluate_average_precision(
            predicted_frames,
            truth_frames,
            iou_threshold=value,
            class_aware=class_aware,
        )
        for value in thresholds
    )
    return MeanAveragePrecisionResult(
        iou_thresholds=thresholds,
        per_threshold=results,
        mean_average_precision=float(
            np.mean([item.average_precision for item in results])
        ),
        mean_average_recall=float(np.mean([item.average_recall for item in results])),
    )


@dataclass(frozen=True)
class TrackingMetrics:
    """Approximate MOT metrics for deterministic regression checks."""

    true_positives: int
    false_positives: int
    false_negatives: int
    identity_switches: int
    ground_truth_objects: int

    @property
    def mota(self) -> float:
        if not self.ground_truth_objects:
            return 0.0
        errors = self.false_positives + self.false_negatives + self.identity_switches
        return 1.0 - errors / self.ground_truth_objects

    @property
    def idf1(self) -> float:
        idtp = self.true_positives
        idfp = self.false_positives + self.identity_switches
        idfn = self.false_negatives + self.identity_switches
        denominator = 2 * idtp + idfp + idfn
        return 2 * idtp / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Union[float, int]]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "identity_switches": self.identity_switches,
            "ground_truth_objects": self.ground_truth_objects,
            "mota": self.mota,
            "idf1": self.idf1,
        }


def evaluate_tracking(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> TrackingMetrics:
    """Evaluate frame sequences with greedy matches and ID-switch counts."""
    predicted_frames = list(predictions)
    truth_frames = list(ground_truth)
    if len(predicted_frames) != len(truth_frames):
        raise ValueError("predictions and ground_truth must have equal frame counts")
    true_positives = false_positives = false_negatives = switches = total = 0
    previous_ids: dict[int, Optional[int]] = {}
    for predicted, expected in zip(predicted_frames, truth_frames):
        predicted_list = sorted(
            list(predicted), key=lambda item: item.score, reverse=True
        )
        expected_list = list(expected)
        total += len(expected_list)
        used: set[int] = set()
        for detection in predicted_list:
            best_index: Optional[int] = None
            best_overlap = iou_threshold
            for index, target in enumerate(expected_list):
                if index in used or (class_aware and detection.label != target.label):
                    continue
                overlap = box_iou(detection.box, target.box)
                if overlap >= best_overlap:
                    best_overlap = overlap
                    best_index = index
            if best_index is None:
                false_positives += 1
                continue
            used.add(best_index)
            true_positives += 1
            target = expected_list[best_index]
            if target.track_id is not None:
                previous = previous_ids.get(target.track_id)
                if previous is not None and previous != detection.track_id:
                    switches += 1
                previous_ids[target.track_id] = detection.track_id
        false_negatives += len(expected_list) - len(used)
    return TrackingMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        identity_switches=switches,
        ground_truth_objects=total,
    )


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return IoU for two binary masks, including empty-mask handling."""
    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("masks must have identical shapes")
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return float(intersection / union) if union else 1.0


@dataclass(frozen=True)
class MaskMetrics:
    """Aggregate instance-mask quality at one IoU threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    mean_iou: float
    iou_threshold: float

    @property
    def f1(self) -> float:
        denominator = (
            2 * self.true_positives + self.false_positives + self.false_negatives
        )
        return 2 * self.true_positives / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Union[float, int]]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "mean_iou": self.mean_iou,
            "iou_threshold": self.iou_threshold,
            "f1": self.f1,
        }


def evaluate_masks(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> MaskMetrics:
    """Match instance masks one-to-one and report mask quality."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    predicted = [item for item in predictions if item.mask is not None]
    expected = [item for item in ground_truth if item.mask is not None]
    used: set[int] = set()
    overlaps: list[float] = []
    true_positives = 0
    for detection in sorted(predicted, key=lambda item: item.score, reverse=True):
        best_index: Optional[int] = None
        best_overlap = iou_threshold
        for index, target in enumerate(expected):
            if index in used or (class_aware and detection.label != target.label):
                continue
            overlap = mask_iou(detection.mask, target.mask)
            if overlap >= best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None:
            used.add(best_index)
            true_positives += 1
            overlaps.append(best_overlap)
    return MaskMetrics(
        true_positives=true_positives,
        false_positives=len(predicted) - true_positives,
        false_negatives=len(expected) - true_positives,
        mean_iou=float(np.mean(overlaps)) if overlaps else 0.0,
        iou_threshold=iou_threshold,
    )


def evaluate_mask_average_precision(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
    recall_thresholds: Optional[Iterable[float]] = None,
) -> AveragePrecisionResult:
    """Compute interpolated AP using instance-mask IoU instead of box IoU."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    thresholds = (
        tuple(np.linspace(0.0, 1.0, 101))
        if recall_thresholds is None
        else tuple(float(value) for value in recall_thresholds)
    )
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("recall_thresholds must be non-empty values in [0, 1]")
    return _average_precision_at_threshold(
        predictions,
        ground_truth,
        iou_threshold,
        class_aware,
        thresholds,
        overlap_function=mask_iou,
        require_masks=True,
    )


def evaluate_mean_mask_average_precision(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_thresholds: Optional[Iterable[float]] = None,
    class_aware: bool = True,
) -> MeanAveragePrecisionResult:
    """Compute instance-mask AP/AR over the standard IoU threshold sweep."""
    thresholds = (
        tuple(round(float(value), 2) for value in np.arange(0.5, 0.96, 0.05))
        if iou_thresholds is None
        else tuple(float(value) for value in iou_thresholds)
    )
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("iou_thresholds must be non-empty values in [0, 1]")
    predicted_frames = _as_detection_frames(predictions)
    truth_frames = _as_detection_frames(ground_truth)
    results = tuple(
        evaluate_mask_average_precision(
            predicted_frames,
            truth_frames,
            iou_threshold=value,
            class_aware=class_aware,
        )
        for value in thresholds
    )
    return MeanAveragePrecisionResult(
        iou_thresholds=thresholds,
        per_threshold=results,
        mean_average_precision=float(
            np.mean([item.average_precision for item in results])
        ),
        mean_average_recall=float(np.mean([item.average_recall for item in results])),
    )


def _maximum_assignment(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return a maximum-weight rectangular assignment using Hungarian steps."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        return []
    rows, columns = values.shape
    transposed = False
    if rows > columns:
        values = values.T
        rows, columns = values.shape
        transposed = True
    u = np.zeros(rows + 1)
    v = np.zeros(columns + 1)
    p = np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = -values[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    p[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = p[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = []
    for column in range(1, columns + 1):
        row = p[column]
        if row and values[row - 1, column - 1] > 0:
            pair = (row - 1, column - 1)
            pairs.append((pair[1], pair[0]) if transposed else pair)
    return pairs


@dataclass(frozen=True)
class HOTAResult:
    """Single-threshold HOTA result with detection/association decomposition."""

    hota: float
    detection_accuracy: float
    association_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    iou_threshold: float

    def to_dict(self) -> dict[str, Union[float, int]]:
        return {
            "hota": self.hota,
            "detection_accuracy": self.detection_accuracy,
            "association_accuracy": self.association_accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "iou_threshold": self.iou_threshold,
        }


@dataclass(frozen=True)
class HOTACurveResult:
    """HOTA aggregated over the standard IoU threshold curve."""

    alphas: tuple[float, ...]
    per_threshold: tuple[HOTAResult, ...]
    mean_hota: float
    mean_detection_accuracy: float
    mean_association_accuracy: float

    def to_dict(self) -> dict[str, object]:
        return {
            "alphas": list(self.alphas),
            "per_threshold": [item.to_dict() for item in self.per_threshold],
            "mean_hota": self.mean_hota,
            "mean_detection_accuracy": self.mean_detection_accuracy,
            "mean_association_accuracy": self.mean_association_accuracy,
        }


def evaluate_hota(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> HOTAResult:
    """Evaluate a single IoU-threshold HOTA decomposition.

    This follows the HOTA detection/association decomposition and uses a
    maximum-IoU bipartite assignment per frame. For the official multi-alpha
    MOTChallenge score, call this function over the desired alpha values and
    average the resulting HOTA values.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    predicted_frames = [list(frame) for frame in predictions]
    truth_frames = [list(frame) for frame in ground_truth]
    if len(predicted_frames) != len(truth_frames):
        raise ValueError("predictions and ground_truth must have equal frame counts")
    matches: list[tuple[int, int]] = []
    pair_counts: dict[tuple[int, int], int] = {}
    gt_counts: dict[int, int] = {}
    pred_counts: dict[int, int] = {}
    true_positives = false_positives = false_negatives = 0
    for predicted, expected in zip(predicted_frames, truth_frames):
        gt_ids = [item.track_id for item in expected]
        pred_ids = [item.track_id for item in predicted]
        for item in expected:
            if item.track_id is not None:
                gt_counts[int(item.track_id)] = gt_counts.get(int(item.track_id), 0) + 1
        for item in predicted:
            if item.track_id is not None:
                pred_counts[int(item.track_id)] = (
                    pred_counts.get(int(item.track_id), 0) + 1
                )
        matrix = np.zeros((len(expected), len(predicted)), dtype=np.float64)
        for gt_index, target in enumerate(expected):
            for pred_index, detection in enumerate(predicted):
                if not class_aware or target.label == detection.label:
                    overlap = box_iou(target.box, detection.box)
                    if overlap >= iou_threshold:
                        matrix[gt_index, pred_index] = overlap
        frame_matches = _maximum_assignment(matrix)
        matched_predictions = {pred_index for _, pred_index in frame_matches}
        matched_truth = {gt_index for gt_index, _ in frame_matches}
        true_positives += len(frame_matches)
        false_positives += len(predicted) - len(matched_predictions)
        false_negatives += len(expected) - len(matched_truth)
        for gt_index, pred_index in frame_matches:
            gt_id = gt_ids[gt_index]
            pred_id = pred_ids[pred_index]
            if gt_id is not None and pred_id is not None:
                key = (int(gt_id), int(pred_id))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                matches.append(key)
    detection_accuracy = (
        true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
        if true_positives or false_positives or false_negatives
        else 0.0
    )
    association_values = []
    for gt_id, pred_id in matches:
        pair = pair_counts[(gt_id, pred_id)]
        denominator = gt_counts.get(gt_id, 0) + pred_counts.get(pred_id, 0) - pair
        association_values.append(pair / denominator if denominator else 0.0)
    association_accuracy = (
        float(np.mean(association_values)) if association_values else 0.0
    )
    hota = float(np.sqrt(detection_accuracy * association_accuracy))
    return HOTAResult(
        hota=hota,
        detection_accuracy=detection_accuracy,
        association_accuracy=association_accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        iou_threshold=iou_threshold,
    )


def evaluate_hota_curve(
    predictions: Iterable[Iterable[Detection]],
    ground_truth: Iterable[Iterable[Detection]],
    alphas: Optional[Iterable[float]] = None,
    class_aware: bool = True,
) -> HOTACurveResult:
    """Evaluate HOTA over multiple IoU thresholds and average the results.

    The default ``0.05..0.95`` grid is the threshold sweep used by the
    MOTChallenge HOTA definition. Inputs are materialized once so generators
    are safe to pass and every threshold evaluates the exact same sequence.
    """
    thresholds = (
        tuple(round(float(value), 2) for value in np.linspace(0.05, 0.95, 19))
        if alphas is None
        else tuple(float(value) for value in alphas)
    )
    if not thresholds:
        raise ValueError("alphas must contain at least one threshold")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("every alpha must be finite and in [0, 1]")
    predicted_frames = [list(frame) for frame in predictions]
    truth_frames = [list(frame) for frame in ground_truth]
    results = tuple(
        evaluate_hota(
            predicted_frames,
            truth_frames,
            iou_threshold=threshold,
            class_aware=class_aware,
        )
        for threshold in thresholds
    )
    return HOTACurveResult(
        alphas=thresholds,
        per_threshold=results,
        mean_hota=float(np.mean([item.hota for item in results])),
        mean_detection_accuracy=float(
            np.mean([item.detection_accuracy for item in results])
        ),
        mean_association_accuracy=float(
            np.mean([item.association_accuracy for item in results])
        ),
    )


__all__ = [
    "AveragePrecisionResult",
    "DetectionMetrics",
    "HOTAResult",
    "HOTACurveResult",
    "MaskMetrics",
    "MeanAveragePrecisionResult",
    "TrackingMetrics",
    "evaluate_hota",
    "evaluate_hota_curve",
    "evaluate_average_precision",
    "evaluate_mask_average_precision",
    "evaluate_mean_mask_average_precision",
    "evaluate_mean_average_precision",
    "evaluate_masks",
    "evaluate_detections",
    "evaluate_tracking",
    "mask_iou",
]
