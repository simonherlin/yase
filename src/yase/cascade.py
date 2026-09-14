"""Budget-aware orchestration of fast and accurate semantic extractors.

The cascade is deliberately model agnostic.  It uses the output confidence,
frame novelty and simple temporal stability signals to decide whether an
expensive extractor should be invoked.  This keeps the core usable on CPU
while allowing applications to attach large open-vocabulary models.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core import ImageInput, SemanticResult, load_image
from .schema import Detection


@dataclass(frozen=True)
class RouteDecision:
    """Explanation of the route selected for one input."""

    route: str
    confidence: float
    novelty: float
    reason: str
    fast_seconds: float
    accurate_seconds: Optional[float] = None


@dataclass(frozen=True)
class CascadePolicy:
    """Thresholds controlling :class:`AdaptiveSemanticCascade`."""

    min_fast_confidence: float = 0.65
    max_novelty: float = 0.18
    stable_frames: int = 3
    accurate_on_first_frame: bool = True
    run_accurate_on_empty: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_fast_confidence <= 1:
            raise ValueError("min_fast_confidence must be in [0, 1]")
        if not 0 <= self.max_novelty <= 1:
            raise ValueError("max_novelty must be in [0, 1]")
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")


def _invoke(extractor: Any, image: np.ndarray) -> Any:
    if hasattr(extractor, "extract"):
        return extractor.extract(image)
    if hasattr(extractor, "predict"):
        return extractor.predict(image)
    if callable(extractor):
        return extractor(image)
    raise TypeError("extractor must expose extract/predict or be callable")


def _coerce(value: Any) -> SemanticResult:
    if isinstance(value, SemanticResult):
        return value
    if isinstance(value, dict):
        return SemanticResult(
            depth=value.get("depth"),
            segmentation=value.get("segmentation", value.get("mask")),
            detections=value.get("detections"),
            tags=value.get("tags"),
            embeddings=value.get("embeddings"),
            ocr=value.get("ocr", value.get("text")),
            caption=value.get("caption"),
            scene=value.get("scene"),
            events=value.get("events"),
            keypoints=value.get("keypoints", value.get("poses")),
            relations=value.get("relations"),
            document=value.get("document"),
            depth_map=value.get("depth_map"),
            metadata={
                key: item
                for key, item in value.items()
                if key
                not in {
                    "depth",
                    "segmentation",
                    "mask",
                    "detections",
                    "tags",
                    "embeddings",
                    "ocr",
                    "text",
                    "caption",
                    "scene",
                    "events",
                }
            },
        )
    raise TypeError("extractor output must be SemanticResult or mapping")


def _confidence(result: SemanticResult) -> float:
    detections = result.detections or []
    scores = []
    for detection in detections:
        if isinstance(detection, Detection):
            scores.append(float(detection.score))
        elif isinstance(detection, dict) and "score" in detection:
            scores.append(float(detection["score"]))
    if scores:
        return float(np.clip(max(scores), 0.0, 1.0))
    if result.tags:
        return 0.5
    if result.caption or result.scene is not None:
        return 0.5
    return 0.0


def _novelty(current: np.ndarray, previous: Optional[np.ndarray]) -> float:
    if previous is None:
        return 1.0
    current_small = current.astype(np.float32)[::8, ::8]
    previous_small = previous.astype(np.float32)[::8, ::8]
    if current_small.shape != previous_small.shape:
        return 1.0
    difference = np.mean(np.abs(current_small - previous_small)) / 255.0
    return float(np.clip(difference, 0.0, 1.0))


class AdaptiveSemanticCascade:
    """Choose a fast or accurate extractor per frame.

    ``fast`` is always called.  ``accurate`` is called only when the fast
    answer is uncertain, the scene changed, or the fast answer is empty.  The
    accurate result replaces only fields that it actually provides, allowing
    the fast backend to remain useful for low-cost fields such as embeddings.
    """

    def __init__(
        self,
        fast: Any,
        accurate: Any,
        policy: Optional[CascadePolicy] = None,
    ) -> None:
        self.fast = fast
        self.accurate = accurate
        self.policy = policy or CascadePolicy()
        self._previous: Optional[np.ndarray] = None
        self._frames = 0
        self._stable = 0

    def reset(self) -> None:
        self._previous = None
        self._frames = 0
        self._stable = 0

    def extract(
        self, image: ImageInput, timestamp: Optional[float] = None
    ) -> SemanticResult:
        array = load_image(image)
        started = time.perf_counter()
        fast_result = _coerce(_invoke(self.fast, array))
        fast_seconds = time.perf_counter() - started
        confidence = _confidence(fast_result)
        novelty = _novelty(array, self._previous)
        empty = not bool(
            fast_result.detections or fast_result.tags or fast_result.caption
        )
        reasons = []
        if self._frames == 0 and self.policy.accurate_on_first_frame:
            reasons.append("first_frame")
        if confidence < self.policy.min_fast_confidence:
            reasons.append("low_confidence")
        if novelty > self.policy.max_novelty:
            reasons.append("scene_change")
        if empty and self.policy.run_accurate_on_empty:
            reasons.append("empty_fast_result")
        should_refine = bool(reasons)
        accurate_seconds = None
        result = fast_result
        route = "fast"
        if should_refine:
            started = time.perf_counter()
            accurate_result = _coerce(_invoke(self.accurate, array))
            accurate_seconds = time.perf_counter() - started
            result = self._merge(fast_result, accurate_result)
            route = "fast+accurate"
            self._stable = 0
        else:
            self._stable += 1
        self._frames += 1
        self._previous = array
        decision = RouteDecision(
            route=route,
            confidence=confidence,
            novelty=novelty,
            reason=",".join(reasons) if reasons else "stable_fast_path",
            fast_seconds=fast_seconds,
            accurate_seconds=accurate_seconds,
        )
        metadata = dict(result.metadata)
        metadata["cascade"] = {
            "route": decision.route,
            "confidence": decision.confidence,
            "novelty": decision.novelty,
            "reason": decision.reason,
            "fast_seconds": decision.fast_seconds,
            "accurate_seconds": decision.accurate_seconds,
            "stable_frames": self._stable,
        }
        return SemanticResult(
            depth=result.depth,
            segmentation=result.segmentation,
            detections=result.detections,
            tags=result.tags,
            embeddings=result.embeddings,
            timestamp=result.timestamp if timestamp is None else timestamp,
            metadata=metadata,
            ocr=result.ocr,
            caption=result.caption,
            scene=result.scene,
            events=result.events,
            keypoints=result.keypoints,
            relations=result.relations,
            document=result.document,
            depth_map=result.depth_map,
        )

    @staticmethod
    def _merge(fast: SemanticResult, accurate: SemanticResult) -> SemanticResult:
        def choose(name: str) -> Any:
            accurate_value = getattr(accurate, name)
            return accurate_value if accurate_value is not None else getattr(fast, name)

        metadata = dict(fast.metadata)
        metadata.update(accurate.metadata)
        metadata["cascade_fast_metadata"] = dict(fast.metadata)
        return SemanticResult(
            depth=choose("depth"),
            segmentation=choose("segmentation"),
            detections=choose("detections"),
            tags=choose("tags"),
            embeddings=choose("embeddings"),
            timestamp=choose("timestamp"),
            metadata=metadata,
            ocr=choose("ocr"),
            caption=choose("caption"),
            scene=choose("scene"),
            events=choose("events"),
            keypoints=choose("keypoints"),
            relations=choose("relations"),
            document=choose("document"),
            depth_map=choose("depth_map"),
        )

    def __call__(
        self, image: ImageInput, timestamp: Optional[float] = None
    ) -> SemanticResult:
        return self.extract(image, timestamp=timestamp)


__all__ = ["AdaptiveSemanticCascade", "CascadePolicy", "RouteDecision"]
