"""Dependency-free global identity matching for multiple camera streams."""

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
