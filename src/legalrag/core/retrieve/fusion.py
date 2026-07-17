"""Rank fusion (master plan §5.3). RRF is the E1 default."""

from __future__ import annotations

from collections.abc import Sequence

from legalrag.core.interfaces import Stage
from legalrag.core.models import ScoredChunk
from legalrag.core.registry import register


@register(Stage.FUSION, "rrf")
class RRFFusion:
    """Reciprocal Rank Fusion. Combines any number of ranked lists (dense,
    sparse, per-subquery) robustly without score normalization."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, ranked_lists: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:
        agg: dict[str, float] = {}
        rep: dict[str, ScoredChunk] = {}
        comps: dict[str, dict[str, float]] = {}
        for ranked in ranked_lists:
            for rank, sc in enumerate(ranked):
                agg[sc.chunk_id] = agg.get(sc.chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
                if sc.chunk_id not in rep:
                    rep[sc.chunk_id] = sc
                    comps[sc.chunk_id] = dict(sc.components)
                else:
                    comps[sc.chunk_id].update(sc.components)
        fused = [
            ScoredChunk(
                chunk_id=cid,
                doc_id=rep[cid].doc_id,
                score=score,
                para_ids=list(rep[cid].para_ids),
                text=rep[cid].text,
                rank=rank,
                components={**comps[cid], "rrf": score},
            )
            for rank, (cid, score) in enumerate(
                sorted(agg.items(), key=lambda x: -x[1])
            )
        ]
        return fused
