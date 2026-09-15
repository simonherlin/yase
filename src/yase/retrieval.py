"""Optional production vector-store integrations."""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from .index import SearchHit
from .observation import EmbeddingRecord


class QdrantVectorIndex:
    """Qdrant-backed counterpart to :class:`NumpyVectorIndex`.

    Pass an existing client in tests or applications. The Qdrant dependency is
    imported only when this adapter is instantiated.
    """

    def __init__(
        self,
        collection: str,
        dimension: Optional[int] = None,
        *,
        client: Optional[Any] = None,
        url: Optional[str] = None,
        path: Optional[str] = None,
        models: Optional[Any] = None,
        space: Optional[str] = None,
    ) -> None:
        if not collection:
            raise ValueError("collection must not be empty")
        if space is not None and not space:
            raise ValueError("space must not be empty when provided")
        if client is None or models is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client import models as qdrant_models
            except ImportError as exc:
                raise ImportError(
                    "install the retrieval extra to use QdrantVectorIndex"
                ) from exc
            models = models or qdrant_models
            if client is None:
                client = QdrantClient(path=path) if path else QdrantClient(url=url)
        self.collection = collection
        self.dimension = dimension
        self.client = client
        self.models = models
        self.space = space
        if self.dimension is not None:
            self._ensure_collection()

    def _ensure_collection(self) -> None:
        exists = self.client.collection_exists(self.collection)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self.models.VectorParams(
                    size=self.dimension, distance=self.models.Distance.COSINE
                ),
            )

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
        if self.dimension is None:
            self.dimension = int(value.size)
            self._ensure_collection()
        if value.size != self.dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}")
        norm = float(np.linalg.norm(value))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("vector must be finite and non-zero")
        payload = dict(metadata or {})
        if self.space is not None:
            payload_space = payload.get("space")
            if payload_space is not None and payload_space != self.space:
                raise ValueError(f"expected embeddings from space {self.space}")
        point = self.models.PointStruct(
            id=item_id,
            vector=(value / norm).tolist(),
            payload=payload,
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    def add_record(
        self,
        item_id: str,
        record: EmbeddingRecord,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Add a provenance-aware embedding and bind the collection to its space."""
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

    def search(
        self,
        vector: Any,
        limit: int = 10,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        value = np.asarray(vector, dtype=np.float32)
        if value.ndim != 1 or value.size != self.dimension:
            raise ValueError(f"expected a vector with dimension {self.dimension}")
        norm = float(np.linalg.norm(value))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("query vector must be finite and non-zero")
        query_filter = None
        if where:
            query_filter = self.models.Filter(
                must=[
                    self.models.FieldCondition(
                        key=key, match=self.models.MatchValue(value=match)
                    )
                    for key, match in where.items()
                ]
            )
        query = (value / norm).tolist()
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        else:
            points = self.client.search(
                collection_name=self.collection,
                query_vector=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        return [
            SearchHit(
                str(point.id),
                float(point.score),
                dict(getattr(point, "payload", None) or {}),
            )
            for point in points
        ]

    def search_record(
        self,
        record: EmbeddingRecord,
        limit: int = 10,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[SearchHit]:
        """Search with a provenance-aware query embedding."""
        if not isinstance(record, EmbeddingRecord):
            raise TypeError("record must be an EmbeddingRecord")
        if self.space is not None and record.space != self.space:
            raise ValueError(f"expected embeddings from space {self.space}")
        return self.search(record.vector, limit=limit, where=where)


__all__ = ["QdrantVectorIndex"]
