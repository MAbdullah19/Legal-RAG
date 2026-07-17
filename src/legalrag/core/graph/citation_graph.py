"""The citation graph and authority scorer (master plan §5.5).

Pure-Python (no networkx) so it stays in the hermetic smoke/CI path per ADR 0003;
research-scale corpora are small enough that adjacency dicts + an iterative
PageRank are ample. NetworkX/Neo4j are the scale options noted in the plan.

Authority score combines four legal-realistic signals:
  authority(d) = pagerank(d) * court_weight(d) * recency(d) * (1 - negative_penalty(d))
- **pagerank**: cases cited more (and by more authoritative cases) rank higher;
- **court_weight**: apex ≻ appellate ≻ trial (from the canonical CourtLevel);
- **recency**: exponential decay so stale authority fades (foundational cases
  still score via inbound citations);
- **negative_penalty**: incoming overruled/distinguished treatments demote a case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from legalrag.core.corpus import Corpus
from legalrag.core.models import NEGATIVE_TREATMENTS, Treatment

_DAMPING = 0.85
_PR_ITERS = 60
_RECENCY_TAU_YEARS = 15.0
_OVERRULED_PENALTY = 0.7
_DISTINGUISHED_PENALTY = 0.15


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    treatment: Treatment | None = None


@dataclass(slots=True)
class CitationGraph:
    nodes: set[str] = field(default_factory=set)
    edges: list[Edge] = field(default_factory=list)
    _out: dict[str, list[Edge]] = field(default_factory=dict)
    _in: dict[str, list[Edge]] = field(default_factory=dict)
    _ref_date: date | None = None
    _authority: dict[str, float] | None = None
    _corpus: Corpus | None = None

    # --- construction ---------------------------------------------------

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> CitationGraph:
        g = cls()
        ref: date | None = None
        for doc in corpus:
            g.nodes.add(doc.doc_id)
            if doc.decision_date and (ref is None or doc.decision_date > ref):
                ref = doc.decision_date
        for doc in corpus:
            for m in doc.citations_out:
                tgt = m.target_doc_id
                if not tgt or tgt not in corpus or tgt == doc.doc_id:
                    continue
                edge = Edge(src=doc.doc_id, dst=tgt, treatment=m.treatment)
                g.edges.append(edge)
                g._out.setdefault(doc.doc_id, []).append(edge)
                g._in.setdefault(tgt, []).append(edge)
        g._ref_date = ref
        g._corpus = corpus
        return g

    # --- adjacency ------------------------------------------------------

    def cites(self, doc_id: str) -> list[str]:
        return [e.dst for e in self._out.get(doc_id, [])]

    def cited_by(self, doc_id: str) -> list[str]:
        return [e.src for e in self._in.get(doc_id, [])]

    def incoming_treatments(self, doc_id: str) -> set[Treatment]:
        return {e.treatment for e in self._in.get(doc_id, []) if e.treatment is not None}

    def is_overruled(self, doc_id: str) -> bool:
        return Treatment.OVERRULED in self.incoming_treatments(doc_id)

    def is_questioned(self, doc_id: str) -> bool:
        """Overruled or distinguished by any citing authority."""
        return bool(self.incoming_treatments(doc_id) & NEGATIVE_TREATMENTS)

    def neighbours(self, doc_id: str) -> list[str]:
        """Unique citing + cited docs, most authoritative first."""
        seen: dict[str, None] = {}
        for nb in self.cited_by(doc_id) + self.cites(doc_id):
            seen.setdefault(nb, None)
        auth = self.authority_scores()
        return sorted(seen, key=lambda d: -auth.get(d, 0.0))

    # --- authority ------------------------------------------------------

    def _pagerank(self) -> dict[str, float]:
        n = len(self.nodes)
        if n == 0:
            return {}
        pr = dict.fromkeys(self.nodes, 1.0 / n)
        out_targets = {d: list({e.dst for e in self._out.get(d, [])}) for d in self.nodes}
        base = (1.0 - _DAMPING) / n
        for _ in range(_PR_ITERS):
            nxt = dict.fromkeys(self.nodes, base)
            dangling = 0.0
            for d in self.nodes:
                tgts = out_targets[d]
                if not tgts:
                    dangling += pr[d]
                    continue
                share = _DAMPING * pr[d] / len(tgts)
                for t in tgts:
                    nxt[t] += share
            if dangling:
                spread = _DAMPING * dangling / n
                for d in self.nodes:
                    nxt[d] += spread
            pr = nxt
        return pr

    def _recency(self, doc_id: str) -> float:
        if self._corpus is None or self._ref_date is None:
            return 1.0
        doc = self._corpus.get(doc_id)
        if doc is None or doc.decision_date is None:
            return 1.0
        age_years = (self._ref_date - doc.decision_date).days / 365.25
        return math.exp(-max(age_years, 0.0) / _RECENCY_TAU_YEARS)

    def _negative_penalty(self, doc_id: str) -> float:
        treatments = self.incoming_treatments(doc_id)
        if Treatment.OVERRULED in treatments:
            return _OVERRULED_PENALTY
        if Treatment.DISTINGUISHED in treatments:
            return _DISTINGUISHED_PENALTY
        return 0.0

    def _court_weight(self, doc_id: str) -> float:
        if self._corpus is None:
            return 0.5
        doc = self._corpus.get(doc_id)
        return doc.authority_weight() if doc is not None else 0.5

    def authority_scores(self) -> dict[str, float]:
        if self._authority is not None:
            return self._authority
        pr = self._pagerank()
        raw: dict[str, float] = {}
        for d in self.nodes:
            raw[d] = (
                pr.get(d, 0.0)
                * self._court_weight(d)
                * self._recency(d)
                * (1.0 - self._negative_penalty(d))
            )
        top = max(raw.values(), default=0.0)
        self._authority = {d: (v / top if top > 0 else 0.0) for d, v in raw.items()}
        return self._authority
