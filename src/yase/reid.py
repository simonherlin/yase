"""Dependency-free global identity matching for multiple camera streams."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

from .schema import Detection


def cosine_similarity(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float32).reshape(-1)
    second = np.asarray(right, dtype=np.float32).reshape(-1)
    if first.shape != second.shape or not first.size:
        return 0.0
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


@dataclass(frozen=True)
class GlobalIdentity:
    """State for one global identity across cameras."""

    global_id: int
    embedding: np.ndarray
    first_seen: float
    last_seen: float
    camera_ids: tuple[str, ...]
    observations: int = 1


class GlobalIdentityStore:
    """Match detection embeddings across independent local streams.

    Embeddings are read from ``Detection.attributes['embedding']``. The store
    only assigns global IDs when an embedding exists; local tracking remains
    valid when no appearance model is available.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        decay: float = 0.9,
        max_age: float = 300.0,
        start_id: int = 1,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if not 0 < decay <= 1 or max_age < 0 or start_id < 0:
            raise ValueError("decay must be in (0, 1], age and ID must be non-negative")
        self.similarity_threshold = similarity_threshold
        self.decay = decay
        self.max_age = max_age
        self._next_id = start_id
        self._identities: dict[int, GlobalIdentity] = {}

    @property
    def identities(self) -> dict[int, GlobalIdentity]:
        return dict(self._identities)

    def reset(self) -> None:
        self._identities.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint of global identities."""
        return {
            "version": 1,
            "next_id": self._next_id,
            "identities": [
                {
                    "global_id": identity.global_id,
                    "embedding": identity.embedding.tolist(),
                    "first_seen": identity.first_seen,
                    "last_seen": identity.last_seen,
                    "camera_ids": list(identity.camera_ids),
                    "observations": identity.observations,
                }
                for identity in sorted(
                    self._identities.values(), key=lambda value: value.global_id
                )
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`."""
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported GlobalIdentityStore state version")
        next_id = state.get("next_id")
        if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 0:
            raise ValueError("identity state next_id must be a non-negative integer")
        values = state.get("identities", ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("identity state identities must be a sequence")
        restored: dict[int, GlobalIdentity] = {}
        for item in values:
            if not isinstance(item, Mapping):
                raise TypeError("each identity state entry must be a mapping")
            global_id = item.get("global_id")
            if (
                isinstance(global_id, bool)
                or not isinstance(global_id, int)
                or global_id < 0
                or global_id in restored
            ):
                raise ValueError(
                    "identity state contains an invalid or duplicate global_id"
                )
            embedding = np.asarray(item.get("embedding", ()), dtype=np.float32).reshape(
                -1
            )
            if not embedding.size or not np.isfinite(embedding).all():
                raise ValueError(
                    "identity state embedding must be finite and non-empty"
                )
            first_seen = float(item.get("first_seen", 0.0))
            last_seen = float(item.get("last_seen", first_seen))
            if not math.isfinite(first_seen) or not math.isfinite(last_seen):
                raise ValueError("identity state timestamps must be finite")
            camera_ids = item.get("camera_ids", ())
            if not isinstance(camera_ids, Sequence) or isinstance(
                camera_ids, (str, bytes)
            ):
                raise TypeError("identity state camera_ids must be a sequence")
            observations = item.get("observations", 1)
            if (
                isinstance(observations, bool)
                or not isinstance(observations, int)
                or observations < 1
            ):
                raise ValueError("identity state observations must be positive")
            restored[global_id] = GlobalIdentity(
                global_id=global_id,
                embedding=embedding.copy(),
                first_seen=first_seen,
                last_seen=last_seen,
                camera_ids=tuple(str(camera_id) for camera_id in camera_ids),
                observations=observations,
            )
        if restored and next_id <= max(restored):
            raise ValueError("identity state next_id must exceed global IDs")
        self._next_id = next_id
        self._identities = restored

    def update(
        self,
        detections: list[Detection],
        camera_id: str = "default",
        timestamp: Optional[float] = None,
    ) -> list[Detection]:
        now = float(0.0 if timestamp is None else timestamp)
        self._expire(now)
        assigned: list[Detection] = []
        used: set[int] = set()
        for detection in detections:
            raw_embedding = detection.attributes.get("embedding")
            if raw_embedding is None:
                assigned.append(detection)
                continue
            embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
            best_id: Optional[int] = None
            best_score = self.similarity_threshold
            for global_id, identity in self._identities.items():
                if global_id in used or identity.embedding.shape != embedding.shape:
                    continue
                score = cosine_similarity(embedding, identity.embedding)
                if score >= best_score:
                    best_score = score
                    best_id = global_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                identity = GlobalIdentity(
                    global_id=best_id,
                    embedding=embedding.copy(),
                    first_seen=now,
                    last_seen=now,
                    camera_ids=(str(camera_id),),
                )
            else:
                previous = self._identities[best_id]
                fused = self.decay * previous.embedding + (1 - self.decay) * embedding
                cameras = tuple(dict.fromkeys((*previous.camera_ids, str(camera_id))))
                identity = replace(
                    previous,
                    embedding=fused,
                    last_seen=now,
                    camera_ids=cameras,
                    observations=previous.observations + 1,
                )
            self._identities[best_id] = identity
            used.add(best_id)
            attributes = dict(detection.attributes)
            attributes.update(
                {
                    "global_id": best_id,
                    "global_identity_score": best_score
                    if identity.observations > 1
                    else 1.0,
                    "camera_id": str(camera_id),
                }
            )
            assigned.append(replace(detection, attributes=attributes))
        return assigned

    def _expire(self, now: float) -> None:
        if self.max_age == 0:
            return
        self._identities = {
            identity_id: identity
            for identity_id, identity in self._identities.items()
            if now - identity.last_seen <= self.max_age
        }


__all__ = ["GlobalIdentity", "GlobalIdentityStore", "cosine_similarity"]
