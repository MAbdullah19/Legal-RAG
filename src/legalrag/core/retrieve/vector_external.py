"""Qdrant vector index (master plan §5.3, §8) — the scale + persistence path.

A drop-in for ``memory`` behind the same VectorIndex Protocol: swapping is a config
change (``vector_index: {name: qdrant, url: ...}``). This is what makes the two-PC
setup work — one host runs Qdrant (docker-compose, infra/) and every machine points
``QDRANT_URL`` at it, so an index built on the RTX 3080 is queryable from the dev
box (see docs/deployment.md).

``qdrant-client`` is imported lazily (``pip install -e ".[stores]"``); the pure
mapping helpers below carry no heavy import, so they are unit-tested without a
running Qdrant. Integration against a live server is validated when the stack is up.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from typing import Any

from legalrag.core.interfaces import Stage
from legalrag.core.models import Chunk, ScoredChunk
from legalrag.core.registry import register

# Payload keys the index owns; a chunk's own payload (court_level/date/doc_type/
# jurisdiction filter fields) is merged alongside these.
_RESERVED = ("chunk_id", "doc_id", "para_ids", "text")


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id (Qdrant point ids must be int/UUID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def payload_for(chunk: Chunk) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "para_ids": list(chunk.para_ids),
        "text": chunk.text,
    }
    for key, value in chunk.payload.items():
        if key not in _RESERVED:
            payload[key] = value
    return payload


def filter_conditions(filters: dict[str, object] | None) -> list[tuple[str, str, object]]:
    """Translate the engine's filter dict into ``(key, op, value)`` triples, where
    op is ``any`` (membership) or ``eq``. Pure — mirrors ``matches_filters`` in the
    in-memory index so both stores filter identically."""
    conds: list[tuple[str, str, object]] = []
    for key, want in (filters or {}).items():
        if isinstance(want, (list, tuple, set)):
            conds.append((key, "any", list(want)))
        else:
            conds.append((key, "eq", want))
    return conds


def _to_qdrant_filter(filters: dict[str, object] | None) -> Any:
    conds = filter_conditions(filters)
    if not conds:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must = [
        FieldCondition(key=key, match=(MatchAny(any=val) if op == "any" else MatchValue(value=val)))
        for key, op, val in conds
    ]
    return Filter(must=must)


@register(Stage.VECTOR_INDEX, "qdrant")
class QdrantVectorIndex:
    def __init__(
        self,
        collection: str = "legalrag",
        url: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        recreate: bool = False,
        timeout: float = 30.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "'qdrant-client' is required for the qdrant index. Install with "
                '`pip install -e ".[stores]"` (kept out of the core/smoke deps by design).'
            ) from e
        url = url or os.environ.get("QDRANT_URL")
        self._client = (
            QdrantClient(url=url, timeout=timeout)
            if url
            else QdrantClient(host=host, port=port, timeout=timeout)
        )
        self.collection = collection
        self._recreate = recreate
        self._ready = False

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        exists = self._client.collection_exists(self.collection)
        if exists and self._recreate:
            self._client.delete_collection(self.collection)
            exists = False
        if not exists:
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._ready = True

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return
        from qdrant_client.models import PointStruct

        if not self._ready:
            self._ensure_collection(len(vectors[0]))
        points = [
            PointStruct(id=_point_id(c.chunk_id), vector=list(v), payload=payload_for(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self.collection, points=points)

    def search(
        self, query_vector: Sequence[float], k: int, filters: dict[str, object] | None = None
    ) -> list[ScoredChunk]:
        response = self._client.query_points(
            self.collection,
            query=list(query_vector),
            limit=k,
            query_filter=_to_qdrant_filter(filters),
            with_payload=True,
        )
        out: list[ScoredChunk] = []
        for rank, pt in enumerate(response.points):
            pl = pt.payload or {}
            score = float(pt.score)
            out.append(
                ScoredChunk(
                    chunk_id=str(pl.get("chunk_id", pt.id)),
                    doc_id=str(pl.get("doc_id", "")),
                    score=score,
                    para_ids=list(pl.get("para_ids") or []),
                    text=str(pl.get("text", "")),
                    rank=rank,
                    components={"dense": score},
                )
            )
        return out
