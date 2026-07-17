"""Graph-expanded retrieval + authority reranking (master plan §5.5, E7).

``none`` is the default for E0-E6. ``citation_neighbors`` does two things after
reranking:
  1. **Authority reranking** — blends each hit's authority score into its
     retrieval score, so binding/leading precedent floats up.
  2. **Neighbour injection** — pulls the citing/cited neighbours of the top hits
     into the candidate set (precedent-chain recall that pure text similarity
     misses — e.g. reaching the case that *overruled* a retrieved one).

It needs only the corpus and the citation graph, both available before indexing,
so neighbours are materialized straight from the canonical documents.
"""

from __future__ import annotations

from collections.abc import Sequence

from legalrag.core.corpus import Corpus
from legalrag.core.graph.citation_graph import CitationGraph
from legalrag.core.interfaces import Stage
from legalrag.core.models import ScoredChunk
from legalrag.core.registry import register


@register(Stage.GRAPH_EXPANDER, "none")
class NoGraphExpansion:
    # accepts the injected corpus/graph (like every graph-expander component) and
    # ignores them, so it is a drop-in default.
    def __init__(self, corpus: Corpus | None = None, graph: CitationGraph | None = None) -> None:
        pass

    def expand(self, query: str, scored: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return list(scored)


@register(Stage.GRAPH_EXPANDER, "citation_neighbors")
class CitationNeighborExpander:
    def __init__(
        self,
        corpus: Corpus,
        graph: CitationGraph | None = None,
        authority_boost: float = 0.0,
        expand_top_n: int = 3,
        neighbours_per_hit: int = 2,
        neighbour_paras: int = 2,
    ) -> None:
        self.corpus = corpus
        self.graph = graph or CitationGraph.from_corpus(corpus)
        # Authority reranking (multiplicative boost, score *= 1 + boost*authority)
        # is OFF by default: on this fixture — and in general — relevance is not
        # the same as authority, so a global authority rerank demotes correct-but-
        # lower-authority answers (a genuine finding; it belongs behind intent
        # routing, E6). Injection below is the safe, always-additive default.
        self.authority_boost = authority_boost
        self.expand_top_n = expand_top_n
        self.neighbours_per_hit = neighbours_per_hit
        self.neighbour_paras = neighbour_paras

    def _neighbour_chunk(self, doc_id: str, score: float, authority: float) -> ScoredChunk | None:
        doc = self.corpus.get(doc_id)
        if doc is None:
            return None
        paras = doc.paragraphs()[: self.neighbour_paras]
        if not paras:
            return None
        return ScoredChunk(
            chunk_id=f"{doc_id}#graph",
            doc_id=doc_id,
            score=score,
            para_ids=[p.pid for p in paras],
            text="\n".join(p.text for p in paras),
            components={"authority": authority, "graph_neighbor": 1.0},
        )

    def expand(self, query: str, scored: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not scored:
            return list(scored)
        auth = self.graph.authority_scores()

        rescored: list[ScoredChunk] = []
        present: set[str] = set()
        for s in scored:
            a = auth.get(s.doc_id, 0.0)
            boosted = s.score * (1.0 + self.authority_boost * a)
            rescored.append(
                s.model_copy(update={"score": boosted, "components": {**s.components, "authority": a}})
            )
            present.add(s.doc_id)
        rescored.sort(key=lambda x: -x.score)

        # Injection is strictly additive: neighbours are scored *below every text
        # hit* (a fraction of the minimum retrieved score, ordered among
        # themselves by authority), so they fill the retrieval budget beyond the
        # text hits — reaching precedent-chain cases (the case a hit relies on, or
        # the case that overruled it) — but can never displace a direct match.
        # The pipeline's `rerank_top_k` sets how many text hits are kept; `final_k`
        # sets the total, so `final_k - rerank_top_k` slots go to graph neighbours.
        min_score = min(s.score for s in rescored)
        injected: list[ScoredChunk] = []
        for s in rescored[: self.expand_top_n]:
            added = 0
            for nb in self.graph.neighbours(s.doc_id):
                if nb in present:
                    continue
                a = auth.get(nb, 0.0)
                chunk = self._neighbour_chunk(nb, min_score * (0.1 + 0.4 * a), a)
                if chunk is None:
                    continue
                present.add(nb)
                injected.append(chunk)
                added += 1
                if added >= self.neighbours_per_hit:
                    break
        injected.sort(key=lambda x: -x.score)

        out = rescored + injected
        for i, s in enumerate(out):
            s.rank = i
        return out
