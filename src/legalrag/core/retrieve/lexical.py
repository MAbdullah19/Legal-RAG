"""BM25 lexical index (master plan §5.3).

Pure-Python Okapi BM25 so the smoke path needs no native deps. Lexical retrieval
is not optional in legal IR: citations, section numbers and terms of art are
exact strings that dense similarity misses (this is the E1 hypothesis). ``bm25s``
or OpenSearch register here later for scale, behind the same LexicalIndex Protocol.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from legalrag.core.interfaces import Stage
from legalrag.core.models import Chunk, ScoredChunk
from legalrag.core.registry import register
from legalrag.core.retrieve.vector import matches_filters
from legalrag.core.text import tokenize


@register(Stage.LEXICAL_INDEX, "bm25")
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avg_len = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            toks = tokenize(chunk.text)
            self._chunks.append(chunk)
            self._tokens.append(toks)
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        total = sum(len(t) for t in self._tokens)
        self._avg_len = total / len(self._tokens) if self._tokens else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        # BM25+ style non-negative idf
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self, query_text: str, k: int, filters: dict[str, object] | None = None
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        q_terms = tokenize(query_text)
        scored: list[tuple[float, int]] = []
        for i, toks in enumerate(self._tokens):
            if not matches_filters(self._chunks[i].payload, filters):
                continue
            if not toks:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1.0))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        out: list[ScoredChunk] = []
        for rank, (score, i) in enumerate(scored[:k]):
            chunk = self._chunks[i]
            out.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(score),
                    para_ids=list(chunk.para_ids),
                    text=chunk.text,
                    rank=rank,
                    components={"sparse": float(score)},
                )
            )
        return out
