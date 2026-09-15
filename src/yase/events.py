"""Small, deterministic temporal event engine for semantic streams."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .core import SemanticResult
from .schema import Detection, SemanticEvent


class EventRule(Protocol):
    def evaluate(
        self, result: SemanticResult, timestamp: float
    ) -> list[SemanticEvent]: ...

    def flush(self, timestamp: float) -> list[SemanticEvent]: ...


@dataclass
class StreamContext:
    """Mutable state shared by temporal stages for one stream."""

    stream_id: str = "default"
    frame_index: int = -1
    timestamp: float | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def update(self, frame_index: int, timestamp: float) -> None:
        self.frame_index = frame_index
        self.timestamp = timestamp


class PresenceRule:
    """Emit enter/exit events for a detected label.

    ``enter_frames`` and ``exit_frames`` provide hysteresis so one bad frame
    does not immediately create an alert or close an event.
    """

    def __init__(
        self,
        label: str,
        *,
        event_label: str | None = None,
        threshold: float = 0.5,
        enter_frames: int = 1,
        exit_frames: int = 1,
    ) -> None:
        if not label:
            raise ValueError("label must not be empty")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if enter_frames < 1 or exit_frames < 1:
            raise ValueError("enter_frames and exit_frames must be >= 1")
        self.label = label
        self.event_label = event_label or label
        self.threshold = threshold
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self._present_frames = 0
        self._missing_frames = 0
        self._active = False
        self._start: float | None = None
        self._track_ids: set[int] = set()

    @staticmethod
    def _detections(result: SemanticResult) -> Iterable[Detection]:
        for detection in result.detections or ():
            if isinstance(detection, Detection):
                yield detection

    def evaluate(self, result: SemanticResult, timestamp: float) -> list[SemanticEvent]:
        matches = [
            detection
            for detection in self._detections(result)
            if detection.label == self.label and detection.score >= self.threshold
        ]
        if matches:
            self._present_frames += 1
            self._missing_frames = 0
            self._track_ids.update(
                detection.track_id
                for detection in matches
                if detection.track_id is not None
            )
        else:
            self._missing_frames += 1
            self._present_frames = 0
        events: list[SemanticEvent] = []
        if not self._active and self._present_frames >= self.enter_frames:
            self._active = True
            self._start = timestamp
            events.append(
                SemanticEvent(
                    self.event_label,
                    max(detection.score for detection in matches),
                    timestamp,
                    track_ids=tuple(sorted(self._track_ids)),
                    metadata={"type": "enter", "source_label": self.label},
                )
            )
        elif self._active and self._missing_frames >= self.exit_frames:
            events.append(self._close(timestamp, "exit"))
        return events

    def _close(self, timestamp: float, event_type: str) -> SemanticEvent:
        event = SemanticEvent(
            self.event_label,
            1.0,
            self._start if self._start is not None else timestamp,
            end=timestamp,
            track_ids=tuple(sorted(self._track_ids)),
            metadata={"type": event_type, "source_label": self.label},
        )
        self._active = False
        self._start = None
        self._track_ids.clear()
        self._missing_frames = 0
        return event

    def flush(self, timestamp: float) -> list[SemanticEvent]:
        return [self._close(timestamp, "flush")] if self._active else []


class DwellRule:
    """Emit once when a tracked object remains visible for a duration."""

    def __init__(
        self,
        label: str,
        min_duration: float,
        *,
        event_label: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not label or min_duration <= 0:
            raise ValueError("label and positive min_duration are required")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.label = label
        self.min_duration = float(min_duration)
        self.event_label = event_label or f"{label}.dwell"
        self.threshold = threshold
        self._started: dict[int, float] = {}
        self._emitted: set[int] = set()

    def evaluate(self, result: SemanticResult, timestamp: float) -> list[SemanticEvent]:
        current: set[int] = set()
        events = []
        for detection in result.detections or ():
            if (
                not isinstance(detection, Detection)
                or detection.label != self.label
                or detection.score < self.threshold
                or detection.track_id is None
            ):
                continue
            track_id = int(detection.track_id)
            current.add(track_id)
            self._started.setdefault(track_id, timestamp)
            if (
                track_id not in self._emitted
                and timestamp - self._started[track_id] >= self.min_duration
            ):
                self._emitted.add(track_id)
                events.append(
                    SemanticEvent(
                        self.event_label,
                        detection.score,
                        self._started[track_id],
                        end=timestamp,
                        track_ids=(track_id,),
                        metadata={
                            "type": "dwell",
                            "duration": timestamp - self._started[track_id],
                            "source_label": self.label,
                        },
                    )
                )
        for track_id in set(self._started) - current:
            self._started.pop(track_id, None)
            self._emitted.discard(track_id)
        return events

    def flush(self, timestamp: float) -> list[SemanticEvent]:
        del timestamp
        self._started.clear()
        self._emitted.clear()
        return []


def _center(detection: Detection) -> tuple[float, float]:
    return (
        (detection.box.x1 + detection.box.x2) / 2.0,
        (detection.box.y1 + detection.box.y2) / 2.0,
    )


def _inside(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test with no geometry dependency."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = current
        x2, y2 = previous
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        previous = current
    return inside


class ZoneRule:
    """Emit enter/exit events when tracked objects cross a polygon zone."""

    def __init__(
        self,
        label: str,
        polygon: Sequence[tuple[float, float]],
        *,
        zone_name: str = "zone",
        threshold: float = 0.5,
        enter_frames: int = 1,
        exit_frames: int = 1,
    ) -> None:
        if not label or len(polygon) < 3:
            raise ValueError(
                "label and a polygon with at least three points are required"
            )
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if enter_frames < 1 or exit_frames < 1:
            raise ValueError("enter_frames and exit_frames must be >= 1")
        self.label = label
        self.polygon = tuple((float(x), float(y)) for x, y in polygon)
        self.zone_name = zone_name
        self.threshold = threshold
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self._inside: dict[int, bool] = {}
        self._present: dict[int, int] = {}
        self._missing: dict[int, int] = {}

    def evaluate(self, result: SemanticResult, timestamp: float) -> list[SemanticEvent]:
        current: dict[int, Detection] = {}
        for detection in result.detections or ():
            if (
                isinstance(detection, Detection)
                and detection.label == self.label
                and detection.score >= self.threshold
                and detection.track_id is not None
            ):
                current[int(detection.track_id)] = detection
        events: list[SemanticEvent] = []
        for track_id, detection in current.items():
            is_inside = _inside(_center(detection), self.polygon)
            was_inside = self._inside.get(track_id, False)
            if is_inside:
                self._present[track_id] = self._present.get(track_id, 0) + 1
                self._missing[track_id] = 0
            else:
                self._missing[track_id] = self._missing.get(track_id, 0) + 1
                self._present[track_id] = 0
            if (
                is_inside
                and not was_inside
                and self._present[track_id] >= self.enter_frames
            ):
                events.append(
                    SemanticEvent(
                        f"{self.zone_name}.enter",
                        detection.score,
                        timestamp,
                        track_ids=(track_id,),
                        metadata={"type": "zone_enter", "zone": self.zone_name},
                    )
                )
            elif (
                was_inside
                and not is_inside
                and self._missing[track_id] >= self.exit_frames
            ):
                events.append(
                    SemanticEvent(
                        f"{self.zone_name}.exit",
                        detection.score,
                        timestamp,
                        track_ids=(track_id,),
                        metadata={"type": "zone_exit", "zone": self.zone_name},
                    )
                )
            self._inside[track_id] = is_inside
        return events

    def flush(self, timestamp: float) -> list[SemanticEvent]:
        events = []
        for track_id, is_inside in list(self._inside.items()):
            if is_inside:
                events.append(
                    SemanticEvent(
                        f"{self.zone_name}.exit",
                        1.0,
                        timestamp,
                        end=timestamp,
                        track_ids=(track_id,),
                        metadata={"type": "zone_flush", "zone": self.zone_name},
                    )
                )
        self._inside.clear()
        self._present.clear()
        self._missing.clear()
        return events


class LineCrossingRule:
    """Emit an event when a tracked object's center crosses a line."""

    def __init__(
        self,
        label: str,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        event_label: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not label or start == end:
            raise ValueError("label and a non-zero line are required")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.label = label
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.event_label = event_label or f"{label}.crossing"
        self.threshold = threshold
        self._side: dict[int, float] = {}

    def _side_value(self, point: tuple[float, float]) -> float:
        ax, ay = self.start
        bx, by = self.end
        return (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)

    def evaluate(self, result: SemanticResult, timestamp: float) -> list[SemanticEvent]:
        events = []
        for detection in result.detections or ():
            if (
                not isinstance(detection, Detection)
                or detection.label != self.label
                or detection.score < self.threshold
                or detection.track_id is None
            ):
                continue
            track_id = int(detection.track_id)
            side = self._side_value(_center(detection))
            previous = self._side.get(track_id)
            if previous is not None and side * previous < 0:
                direction = (
                    "positive_to_negative" if previous > 0 else "negative_to_positive"
                )
                events.append(
                    SemanticEvent(
                        self.event_label,
                        detection.score,
                        timestamp,
                        track_ids=(track_id,),
                        metadata={"type": "line_crossing", "direction": direction},
                    )
                )
            self._side[track_id] = side
        return events

    def flush(self, timestamp: float) -> list[SemanticEvent]:
        del timestamp
        self._side.clear()
        return []


class EventEngine:
    """Evaluate a set of event rules and keep their stream-local state."""

    def __init__(self, rules: Iterable[EventRule]) -> None:
        self.rules = list(rules)
        if not self.rules:
            raise ValueError("at least one event rule is required")

    def update(self, result: SemanticResult, timestamp: float) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for rule in self.rules:
            events.extend(rule.evaluate(result, timestamp))
        return events

    def flush(self, timestamp: float) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for rule in self.rules:
            events.extend(rule.flush(timestamp))
        return events


__all__ = [
    "EventEngine",
    "EventRule",
    "DwellRule",
    "LineCrossingRule",
    "PresenceRule",
    "StreamContext",
    "ZoneRule",
]
