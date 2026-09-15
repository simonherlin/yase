"""Model-agnostic multimodal evidence fusion with abstention."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import TemperatureScaler
from .core import SemanticResult
from .schema import BoundingBox, Detection
from .tracking import box_iou


@dataclass(frozen=True)
class FusedEvidence:
    """Auditable evidence behind one fused detection."""

    label: str
    score: float
    sources: tuple[str, ...]
    agreement: float
    abstained: bool = False


class MultimodalConsensus:
    """Fuse detector outputs and expose uncertainty instead of hallucinating.

    Detections are grouped by label and overlapping boxes. Scores are combined
    with a weighted mean, while geometry is averaged by source confidence.
    The result includes an evidence ledger in ``SemanticResult.metadata``.
    """

    def __init__(
        self,
        source_weights: Mapping[str, float] | None = None,
        calibrators: Mapping[str, TemperatureScaler] | None = None,
        iou_threshold: float = 0.35,
        min_score: float = 0.5,
        min_margin: float = 0.05,
    ) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= min_score <= 1 or not 0 <= min_margin <= 1:
            raise ValueError("score thresholds must be in [0, 1]")
        self.source_weights = dict(source_weights or {})
        self.calibrators = dict(calibrators or {})
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.min_margin = min_margin

    def fuse(
        self,
        results: Mapping[str, SemanticResult] | Iterable[tuple[str, SemanticResult]],
        timestamp: float | None = None,
    ) -> SemanticResult:
        items = list(results.items()) if isinstance(results, Mapping) else list(results)
        groups: list[list[tuple[str, Detection, float]]] = []
        for source, result in items:
            for detection in result.detections or []:
                if not isinstance(detection, Detection):
                    continue
                matched = None
                calibrated = self.calibrators.get(source)
                score = (
                    calibrated.transform(detection.score)
                    if calibrated
                    else detection.score
                )
                for group in groups:
                    reference = group[0][1]
                    if (
                        reference.label == detection.label
                        and box_iou(reference.box, detection.box) >= self.iou_threshold
                    ):
                        matched = group
                        break
                if matched is None:
                    groups.append([(source, detection, score)])
                else:
                    matched.append((source, detection, score))

        fused: list[Detection] = []
        ledger: list[dict[str, Any]] = []
        for group in groups:
            weights = [self.source_weights.get(source, 1.0) for source, _, _ in group]
            scores = [score for _, _, score in group]
            weighted_scores = [score * weight for score, weight in zip(scores, weights)]
            total_weight = sum(weights) or 1.0
            score = float(np.clip(sum(weighted_scores) / total_weight, 0.0, 1.0))
            agreement = float(
                len(set(source for source, _, _ in group)) / max(1, len(items))
            )
            labels = sorted({source for source, _, _ in group})
            best = max(score for score in scores)
            abstained = score < self.min_score
            if not abstained and len(items) > 1 and best - score > self.min_margin:
                abstained = True
            box = self._weighted_box(group, weights)
            first = group[0][1]
            attributes = dict(first.attributes)
            attributes.update(
                {
                    "evidence_sources": labels,
                    "evidence_count": len(group),
                    "evidence_agreement": agreement,
                    "abstained": abstained,
                }
            )
            if not abstained:
                fused.append(
                    Detection(
                        label=first.label,
                        score=score,
                        box=box,
                        mask=first.mask,
                        track_id=first.track_id,
                        attributes=attributes,
                    )
                )
            ledger.append(
                {
                    "label": first.label,
                    "score": score,
                    "sources": labels,
                    "agreement": agreement,
                    "abstained": abstained,
                }
            )

        metadata: dict[str, Any] = {"consensus": ledger}
        tags: list[Any] = []
        ocr: list[Any] = []
        embeddings = []
        for _, result in items:
            tags.extend(result.tags or [])
            ocr.extend(result.ocr or [])
            if result.embeddings is not None:
                embeddings.append(np.asarray(result.embeddings, dtype=np.float32))
        embedding = None
        if embeddings:
            shapes = {item.shape for item in embeddings}
            if len(shapes) == 1:
                embedding = np.mean(np.stack(embeddings), axis=0)
                norm = np.linalg.norm(embedding)
                if norm:
                    embedding = embedding / norm
        first_result = items[0][1] if items else SemanticResult()
        return SemanticResult(
            detections=fused,
            tags=list(dict.fromkeys(tags)),
            embeddings=embedding,
            ocr=ocr or None,
            caption=first_result.caption,
            scene=first_result.scene,
            keypoints=first_result.keypoints,
            relations=first_result.relations,
            document=first_result.document,
            depth_map=first_result.depth_map,
            timestamp=timestamp if timestamp is not None else first_result.timestamp,
            metadata=metadata,
        )

    @staticmethod
    def _weighted_box(
        group: list[tuple[str, Detection, float]], weights: list[float]
    ) -> BoundingBox:
        values = []
        for _, detection, _ in group:
            values.append(np.asarray(detection.box.as_xyxy(), dtype=np.float64))
        coordinates = np.average(np.stack(values), axis=0, weights=weights)
        return BoundingBox.from_sequence(coordinates.tolist())

    def __call__(self, results: Mapping[str, SemanticResult]) -> SemanticResult:
        return self.fuse(results)


__all__ = ["FusedEvidence", "MultimodalConsensus"]
