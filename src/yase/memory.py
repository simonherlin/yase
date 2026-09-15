"""Temporal semantic memory for tracked objects."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .core import SemanticResult
from .schema import Detection


@dataclass(frozen=True)
class TrackMemoryState:
    """Stable metadata accumulated for one track identifier."""

    track_id: int
    first_seen: float
    last_seen: float
    seen_count: int
    label: str
    score_ema: float
    labels: tuple[str, ...]


class SemanticTrackMemory:
    """Maintain labels, confidence and lifetime information per track.

    Tracking and memory are separate on purpose: ByteTrack, BoT-SORT, a
    hardware tracker, or the dependency-free IoU tracker can all feed this
    class through the common ``Detection.track_id`` field.
    """

    def __init__(self, decay: float = 0.8, max_missed: int = 30) -> None:
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")
        self.decay = decay
        self.max_missed = max_missed
        self._states: dict[int, TrackMemoryState] = {}
        self._missed: dict[int, int] = {}
        self._frame = 0

    @property
    def states(self) -> dict[int, TrackMemoryState]:
        return dict(self._states)

    def reset(self) -> None:
        self._states.clear()
        self._missed.clear()
        self._frame = 0

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint of semantic track memory."""
        return {
            "version": 1,
            "frame": self._frame,
            "states": [
                {
                    "track_id": state.track_id,
                    "first_seen": state.first_seen,
                    "last_seen": state.last_seen,
                    "seen_count": state.seen_count,
                    "label": state.label,
                    "score_ema": state.score_ema,
                    "labels": list(state.labels),
                    "missed": self._missed[state.track_id],
                }
                for state in sorted(
                    self._states.values(), key=lambda value: value.track_id
                )
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`."""
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported SemanticTrackMemory state version")
        frame = state.get("frame", 0)
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("memory state frame must be a non-negative integer")
        values = state.get("states", ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("memory state states must be a sequence")
        restored: dict[int, TrackMemoryState] = {}
        missed_values: dict[int, int] = {}
        for item in values:
            if not isinstance(item, Mapping):
                raise TypeError("each memory state entry must be a mapping")
            track_id = item.get("track_id")
            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id in restored
            ):
                raise ValueError(
                    "memory state contains an invalid or duplicate track_id"
                )
            label = item.get("label")
            labels = item.get("labels", ())
            if not isinstance(label, str) or not label:
                raise ValueError("memory state labels must be non-empty strings")
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
                raise TypeError("memory state labels must be a sequence")
            if any(not isinstance(value, str) or not value for value in labels):
                raise ValueError("memory state labels must be non-empty strings")
            seen_count = item.get("seen_count", 1)
            missed = item.get("missed", 0)
            if (
                isinstance(seen_count, bool)
                or not isinstance(seen_count, int)
                or seen_count < 1
                or isinstance(missed, bool)
                or not isinstance(missed, int)
                or missed < 0
            ):
                raise ValueError("memory state counters are invalid")
            first_seen = float(item.get("first_seen", 0.0))
            last_seen = float(item.get("last_seen", first_seen))
            score_ema = float(item.get("score_ema", 0.0))
            if (
                not math.isfinite(first_seen)
                or not math.isfinite(last_seen)
                or not math.isfinite(score_ema)
                or not 0 <= score_ema <= 1
            ):
                raise ValueError("memory state values must be finite and valid")
            restored[track_id] = TrackMemoryState(
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                seen_count=seen_count,
                label=label,
                score_ema=score_ema,
                labels=tuple(labels),
            )
            missed_values[track_id] = missed
        self._frame = frame
        self._states = restored
        self._missed = missed_values

    def update(
        self, result: SemanticResult, timestamp: float | None = None
    ) -> SemanticResult:
        self._frame += 1
        now = float(self._frame if timestamp is None else timestamp)
        detections = []
        seen: set[int] = set()
        for detection in result.detections or []:
            if not isinstance(detection, Detection) or detection.track_id is None:
                detections.append(detection)
                continue
            track_id = int(detection.track_id)
            seen.add(track_id)
            previous = self._states.get(track_id)
            if previous is None:
                state = TrackMemoryState(
                    track_id=track_id,
                    first_seen=now,
                    last_seen=now,
                    seen_count=1,
                    label=detection.label,
                    score_ema=detection.score,
                    labels=(detection.label,),
                )
            else:
                score = (
                    self.decay * previous.score_ema + (1 - self.decay) * detection.score
                )
                labels = list(previous.labels)
                labels.append(detection.label)
                state = replace(
                    previous,
                    last_seen=now,
                    seen_count=previous.seen_count + 1,
                    label=Counter(labels).most_common(1)[0][0],
                    score_ema=score,
                    labels=tuple(labels[-32:]),
                )
            self._states[track_id] = state
            self._missed[track_id] = 0
            attributes = dict(detection.attributes)
            attributes.update(
                {
                    "memory_label": state.label,
                    "memory_score": state.score_ema,
                    "first_seen": state.first_seen,
                    "last_seen": state.last_seen,
                    "seen_count": state.seen_count,
                }
            )
            detections.append(replace(detection, attributes=attributes))
        for track_id in list(self._states):
            if track_id not in seen:
                self._missed[track_id] += 1
                if self._missed[track_id] > self.max_missed:
                    del self._states[track_id]
                    del self._missed[track_id]
        metadata = dict(result.metadata)
        metadata["track_memory"] = {
            str(track_id): {
                "label": state.label,
                "score": state.score_ema,
                "seen_count": state.seen_count,
                "missed": self._missed[track_id],
            }
            for track_id, state in self._states.items()
        }
        return SemanticResult(
            depth=result.depth,
            segmentation=result.segmentation,
            detections=detections,
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


__all__ = ["SemanticTrackMemory", "TrackMemoryState"]
