"""Cross-encoder rerankers (master plan §5.3, rung E2) — the local, no-API path.

Kept out of ``rerankers.py`` (which holds the deterministic ``none`` reranker used
by the hermetic smoke pipeline) so importing the smoke path never pulls torch.
Heavy deps are imported **lazily**: registering the component costs nothing, and
instantiating one without ``.[embeddings]`` installed raises a clear, actionable
error (same contract as ``embedders_external``). Default model runs fully locally
on the researcher's machine — no document text leaves the box (ADR 0005).

Install: ``pip install -e ".[embeddings]"`` (sentence-transformers pulls torch).
"""

from __future__ import annotations

from collections.abc import Sequence

from legalrag.core.interfaces import Stage
from legalrag.core.models import ScoredChunk
from legalrag.core.registry import register


@register(Stage.RERANKER, "bge_reranker")
class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder — the E2 contender.

    Scores each (query, chunk-text) pair jointly (unlike the bi-encoder embedder),
    then keeps the ``top_k`` highest. Multilingual (covers the Urdu path for PK).
    Runs locally via sentence-transformers ``CrossEncoder``; first use downloads
    the model to the HF cache, thereafter offline.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "'sentence-transformers' is required for the bge_reranker. Install with "
                '`pip install -e ".[embeddings]"` (kept out of the core/smoke deps by design).'
            ) from e
        self._model_name = model
        self._batch_size = batch_size
        self._ce = CrossEncoder(model, max_length=max_length)

    @property
    def model_id(self) -> str:
        return f"ce:{self._model_name}"

    def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [(query, sc.text) for sc in chunks]
        scores = self._ce.predict(pairs, batch_size=self._batch_size)
        order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        out: list[ScoredChunk] = []
        for rank, i in enumerate(order[:top_k]):
            sc = chunks[i]
            sc.score = float(scores[i])
            sc.rank = rank
            out.append(sc)
        return out
