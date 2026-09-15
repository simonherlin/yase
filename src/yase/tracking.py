"""Dependency-free IoU multi-object tracking baseline."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .native import iou_matrix
from .schema import BoundingBox, Detection


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    """Return intersection-over-union for two axis-aligned boxes."""
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


@dataclass
class _TrackState:
    box: BoundingBox
    label: str
    score: float
    missed: int = 0
    age: int = 1


class IoUTracker:
    """Simple online tracker suitable as a CPU fallback and test oracle.

    It intentionally exposes the same input/output shape as detector adapters.
    For difficult scenes, applications can replace it with ByteTrack or
    BoT-SORT without changing ``SemanticResult``.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 30,
        class_aware: bool = True,
        start_id: int = 1,
    ) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")
        if start_id < 0:
            raise ValueError("start_id must be >= 0")
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.class_aware = class_aware
        self._next_id = start_id
        self._tracks: dict[int, _TrackState] = {}

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        detections: Optional[Sequence[Detection]],
        timestamp: Optional[float] = None,
    ) -> list[Detection]:
        del timestamp  # Reserved for motion-aware trackers with the same API.
        current = list(detections or [])
        candidates = set(self._tracks)
        track_ids = list(candidates)
        overlaps = iou_matrix(
            [self._tracks[track_id].box for track_id in track_ids],
            [detection.box for detection in current],
        )
        assigned: list[Detection] = []
        for detection_index, detection in enumerate(current):
            best_id: Optional[int] = None
            best_iou = self.iou_threshold
            for track_index, track_id in enumerate(track_ids):
                if track_id not in candidates:
                    continue
                state = self._tracks[track_id]
                if self.class_aware and state.label != detection.label:
                    continue
                score = overlaps[track_index][detection_index]
                if score >= best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = _TrackState(
                    detection.box, detection.label, detection.score
                )
            else:
                candidates.remove(best_id)
                state = self._tracks[best_id]
                state.box = detection.box
                state.label = detection.label
                state.score = detection.score
                state.missed = 0
                state.age += 1
            assigned.append(
                Detection(
                    label=detection.label,
                    score=detection.score,
                    box=detection.box,
                    mask=detection.mask,
                    track_id=best_id,
                    attributes=detection.attributes,
                )
            )
        for track_id in candidates:
            self._tracks[track_id].missed += 1
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if state.missed <= self.max_missed
        }
        return assigned


@dataclass
class _MotionTrackState:
    box: BoundingBox
    label: str
    score: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    missed: int = 0
    age: int = 1

    def predict(self) -> BoundingBox:
        return BoundingBox(
            self.box.x1 + self.velocity_x,
            self.box.y1 + self.velocity_y,
            self.box.x2 + self.velocity_x,
            self.box.y2 + self.velocity_y,
        )


class ByteTrackLite:
    """Dependency-free two-stage tracker inspired by ByteTrack.

    High-confidence detections create tracks. Lower-confidence detections are
    used only to recover existing tracks, which helps through partial
    occlusions without allowing isolated noise to create new identities.
    ``ByteTrackLite`` is intentionally small; applications can swap in the
    official ByteTrack or BoT-SORT adapter without changing result schemas.
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.2,
        max_missed: int = 30,
        class_aware: bool = True,
        start_id: int = 1,
    ) -> None:
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed < 0 or start_id < 0:
            raise ValueError("max_missed and start_id must be non-negative")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.class_aware = class_aware
        self._next_id = start_id
        self._tracks: dict[int, _MotionTrackState] = {}

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        detections: Optional[Sequence[Detection]],
        timestamp: Optional[float] = None,
    ) -> list[Detection]:
        del timestamp
        current = list(detections or [])
        high = [
            (index, detection)
            for index, detection in enumerate(current)
            if detection.score >= self.high_threshold
        ]
        low = [
            (index, detection)
            for index, detection in enumerate(current)
            if self.low_threshold <= detection.score < self.high_threshold
        ]
        candidates = set(self._tracks)
        assignments: dict[int, int] = {}

        def match(pool: list[tuple[int, Detection]]) -> None:
            for index, detection in sorted(
                pool, key=lambda item: item[1].score, reverse=True
            ):
                best_id: Optional[int] = None
                best_iou = self.iou_threshold
                for track_id in candidates:
                    state = self._tracks[track_id]
                    if self.class_aware and state.label != detection.label:
                        continue
                    overlap = box_iou(state.predict(), detection.box)
                    if overlap >= best_iou:
                        best_iou = overlap
                        best_id = track_id
                if best_id is not None:
                    candidates.remove(best_id)
                    assignments[index] = best_id

        match(high)
        match(low)
        output: list[Detection] = []
        for index, detection in enumerate(current):
            track_id = assignments.get(index)
            if track_id is None:
                if detection.score < self.high_threshold:
                    continue
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _MotionTrackState(
                    detection.box, detection.label, detection.score
                )
            else:
                state = self._tracks[track_id]
                old_center = (
                    (state.box.x1 + state.box.x2) / 2,
                    (state.box.y1 + state.box.y2) / 2,
                )
                new_center = (
                    (detection.box.x1 + detection.box.x2) / 2,
                    (detection.box.y1 + detection.box.y2) / 2,
                )
                state.velocity_x = new_center[0] - old_center[0]
                state.velocity_y = new_center[1] - old_center[1]
                state.box = detection.box
                state.label = detection.label
                state.score = detection.score
                state.missed = 0
                state.age += 1
            output.append(
                Detection(
                    label=detection.label,
                    score=detection.score,
                    box=detection.box,
                    mask=detection.mask,
                    track_id=track_id,
                    attributes={**detection.attributes, "tracker": "bytetrack-lite"},
                )
            )
        for track_id in candidates:
            state = self._tracks[track_id]
            state.box = state.predict()
            state.missed += 1
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if state.missed <= self.max_missed
        }
        return output


class ExternalTrackerAdapter:
    """Normalize an external ByteTrack/BoT-SORT-style tracker.

    The external object may expose ``update(detections)`` or
    ``update(boxes, scores, labels)``. A custom ``converter`` can normalize
    framework-specific outputs; without one, lists of ``Detection`` or
    mappings containing ``box``/``track_id`` are supported.
    """

    def __init__(
        self,
        tracker: Any,
        converter: Optional[Callable[[Any], Sequence[Detection]]] = None,
    ) -> None:
        if not hasattr(tracker, "update") and not hasattr(tracker, "track"):
            raise TypeError("tracker must expose update or track")
        self.tracker = tracker
        self.converter = converter

    def update(
        self,
        detections: Optional[Sequence[Detection]],
        timestamp: Optional[float] = None,
    ) -> list[Detection]:
        current = list(detections or [])
        method = getattr(self.tracker, "update", None) or self.tracker.track
        try:
            raw = method(current, timestamp=timestamp)
        except TypeError:
            try:
                raw = method(current)
            except TypeError:
                raw = method(
                    [item.box.as_xyxy() for item in current],
                    [item.score for item in current],
                    [item.label for item in current],
                )
        if self.converter is not None:
            return list(self.converter(raw))
        if raw is None:
            return []
        output: list[Detection] = []
        for index, value in enumerate(raw):
            if isinstance(value, Detection):
                output.append(value)
                continue
            if not isinstance(value, Mapping):
                raise TypeError("external tracker output needs a converter")
            source = current[index] if index < len(current) else None
            box = value.get("box", value.get("bbox"))
            if box is None:
                raise ValueError("external tracker mapping has no box/bbox")
            output.append(
                Detection(
                    label=str(value.get("label", source.label if source else "object")),
                    score=float(value.get("score", source.score if source else 1.0)),
                    box=box,
                    mask=source.mask if source else None,
                    track_id=(
                        int(value["track_id"])
                        if value.get("track_id") is not None
                        else None
                    ),
                    attributes=dict(value.get("attributes", {})),
                )
            )
        return output


__all__ = ["ByteTrackLite", "ExternalTrackerAdapter", "IoUTracker", "box_iou"]
