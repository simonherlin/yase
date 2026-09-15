"""Small local vector index for semantic retrieval.

This is intentionally a dependable NumPy baseline. Production applications can
replace it with Qdrant, FAISS, Milvus, or another vector service without
changing the extraction API.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .core import SemanticResult
from .observation import EmbeddingRecord


@dataclass(frozen=True)
class SearchHit:
    """One cosine-similarity result."""

    item_id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class NumpyVectorIndex:
    """In-memory cosine index for image or video embeddings.

    It is useful for notebooks, tests, edge deployments, and small collections.
    It has deliberately no persistence or server dependency; use a vector
    database adapter when the collection or concurrency requirements grow.
    """

    def __init__(
        self, dimension: Optional[int] = None, space: Optional[str] = None
    ) -> None:
        if space is not None and not space:
            raise ValueError("space must not be empty when provided")
        self.dimension = dimension
        self.space = space
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Mapping[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def add(
        self,
        item_id: str,
        vector: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not item_id:
            raise ValueError("item_id must not be empty")
        value = np.asarray(vector, dtype=np.float32)
        if value.ndim != 1 or value.size == 0:
            raise ValueError("vector must be a non-empty one-dimensional array")
        if not np.isfinite(value).all():
            raise ValueError("vector must contain only finite values")
        norm = float(np.linalg.norm(value))
        if norm == 0:
            raise ValueError("vector must not be all zeros")
        if self.dimension is None:
            self.dimension = int(value.size)
        if value.size != self.dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}")
        self._vectors[item_id] = value / norm
        self._metadata[item_id] = dict(metadata or {})

    def add_result(
        self,
        item_id: str,
        result: SemanticResult,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if result.embeddings is None:
            raise ValueError("SemanticResult does not contain embeddings")
        self.add(item_id, result.embeddings, metadata=metadata)

    def add_record(
        self,
        item_id: str,
        record: EmbeddingRecord,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Add a provenance-aware embedding and bind the index to its space."""
        if not isinstance(record, EmbeddingRecord):
            raise TypeError("record must be an EmbeddingRecord")
        if self.space is None:
            self.space = record.space
        if record.space != self.space:
            raise ValueError(f"expected embeddings from space {self.space}")
        payload = dict(metadata or {})
        payload.setdefault("space", record.space)
        payload.setdefault("model_id", record.model_id)
        if record.revision is not None:
            payload.setdefault("revision", record.revision)
        self.add(item_id, record.vector, metadata=payload)

    def remove(self, item_id: str) -> None:
        self._vectors.pop(item_id, None)
        self._metadata.pop(item_id, None)

    def search(
        self,
        vector: Any,
        limit: int = 10,
        min_score: Optional[float] = None,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if not self._vectors:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.size != self.dimension:
            raise ValueError(f"expected a vector with dimension {self.dimension}")
        norm = float(np.linalg.norm(query))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("query vector must be finite and non-zero")
        query = query / norm
        ids = list(self._vectors)
        matrix = np.stack([self._vectors[item_id] for item_id in ids])
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")[:limit]
        hits = []
        for index in order:
            score = float(scores[index])
            item_id = ids[int(index)]
            if min_score is not None and score < min_score:
                continue
            if where is not None and any(
                self._metadata[item_id].get(key) != value
                for key, value in where.items()
            ):
                continue
            hits.append(SearchHit(item_id, score, self._metadata[item_id]))
        return hits

    def search_record(
        self,
        record: EmbeddingRecord,
        limit: int = 10,
        min_score: Optional[float] = None,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[SearchHit]:
        """Search with a provenance-aware query embedding."""
        if not isinstance(record, EmbeddingRecord):
            raise TypeError("record must be an EmbeddingRecord")
        if self.space is not None and record.space != self.space:
            raise ValueError(f"expected embeddings from space {self.space}")
        return self.search(record.vector, limit=limit, min_score=min_score, where=where)

    def save(self, destination: Any) -> None:
        """Persist vectors and JSON-compatible metadata to a compressed NPZ."""
        path = Path(destination)
        ids = list(self._vectors)
        vectors = (
            np.stack([self._vectors[item_id] for item_id in ids])
            if ids
            else np.empty((0, self.dimension or 0), dtype=np.float32)
        )
        metadata = json.dumps(
            {item_id: self._metadata[item_id] for item_id in ids}, ensure_ascii=False
        )
        np.savez_compressed(
            path,
            ids=np.asarray(ids, dtype=str),
            vectors=vectors,
            metadata=np.asarray(metadata),
            space=np.asarray(self.space or ""),
        )

    @classmethod
    def load(cls, source: Any) -> "NumpyVectorIndex":
        """Load an index created by :meth:`save`."""
        with np.load(Path(source), allow_pickle=False) as archive:
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
            ids = [str(value) for value in archive["ids"].tolist()]
            metadata = json.loads(str(archive["metadata"].item()))
            space = str(archive["space"].item()) if "space" in archive else ""
        dimension = (
            int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[1] else None
        )
        index = cls(dimension=dimension, space=space or None)
        for item_id, vector in zip(ids, vectors):
            index.add(item_id, vector, metadata=metadata.get(item_id, {}))
        return index


__all__ = ["NumpyVectorIndex", "SearchHit"]
