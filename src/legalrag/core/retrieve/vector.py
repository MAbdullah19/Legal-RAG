"""In-memory dense vector index (master plan §5.3).

Cosine similarity over L2-normalized vectors (dot product). Payload filters are
applied pre-search, mirroring Qdrant's filtered search so the ``qdrant`` index
(added later) is a drop-in swap behind the same VectorIndex Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from legalrag.core.interfaces import Stage
from legalrag.core.models import Chunk, ScoredChunk
from legalrag.core.registry import register


def matches_filters(payload: dict[str, object], filters: dict[str, object] | None) -> bool:
    if not filters:
        return True
    for key, want in filters.items():
        have = payload.get(key)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


@register(Stage.VECTOR_INDEX, "memory")
class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return
        new = np.asarray(vectors, dtype=np.float64)
        self._chunks.extend(chunks)
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def search(
        self, query_vector: Sequence[float], k: int, filters: dict[str, object] | None = None
    ) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float64)
        scores = self._matrix @ q  # normalized -> cosine
        order = np.argsort(-scores)
        out: list[ScoredChunk] = []
        for rank_idx in order:
            chunk = self._chunks[int(rank_idx)]
            if not matches_filters(chunk.payload, filters):
                continue
            out.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(scores[int(rank_idx)]),
                    para_ids=list(chunk.para_ids),
                    text=chunk.text,
                    rank=len(out),
                    components={"dense": float(scores[int(rank_idx)])},
                )
            )
            if len(out) >= k:
                break
        return out
