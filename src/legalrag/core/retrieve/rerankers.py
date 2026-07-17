"""Rerankers (master plan §5.3). E0/E1 use ``none``; the cross-encoder
(BGE-reranker-v2-m3) is the E2 contender and registers here later."""

from __future__ import annotations

from collections.abc import Sequence

from legalrag.core.interfaces import Stage
from legalrag.core.models import ScoredChunk
from legalrag.core.registry import register


@register(Stage.RERANKER, "none")
class IdentityReranker:
    def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        out = list(chunks[:top_k])
        for i, sc in enumerate(out):
            sc.rank = i
        return out
