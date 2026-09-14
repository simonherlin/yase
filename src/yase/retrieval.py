"""Optional production vector-store integrations."""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from .index import SearchHit


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
    ) -> None:
        if not collection:
            raise ValueError("collection must not be empty")
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
        value = np.asarray(vector, dtype=np.float32)
        if value.ndim != 1 or value.size == 0:
            raise ValueError("vector must be a non-empty one-dimensional array")
        if self.dimension is None:
            self.dimension = int(value.size)
            self._ensure_collection()
        if value.size != self.dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}")
        norm = float(np.linalg.norm(value))
        if norm == 0:
            raise ValueError("vector must not be all zeros")
        point = self.models.PointStruct(
            id=item_id,
            vector=(value / norm).tolist(),
            payload=dict(metadata or {}),
        )
        self.client.upsert(collection_name=self.collection, points=[point])

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
        if norm == 0:
            raise ValueError("query vector must not be all zeros")
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


__all__ = ["QdrantVectorIndex"]
