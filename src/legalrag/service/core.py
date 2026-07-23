"""Framework-neutral service layer over :class:`Pipeline`.

This is the one place that turns the research pipeline into a *served knowledge
base*: it loads an experiment's pipeline **once** (building the index at
startup, not per request) and exposes two stable operations —

- :meth:`LegalRAGService.answer` — a grounded, citation-verified answer, and
- :meth:`LegalRAGService.search` — ranked source passages (retrieval only).

Both the MCP server (``mcp_server.py``) and the HTTP API (``api.py``) are thin
adapters over this module and share the JSON-serializable response models below,
so the two front doors can never drift in what they return.

The pipeline's components are not proven thread-safe (the in-memory indices and
a local-LLM generator carry mutable state), so calls are serialized with a lock.
That is fine for a single-node knowledge-base server; scale-out would run
multiple workers, each with its own :class:`LegalRAGService`.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, Field

from legalrag.core.config import load_experiment
from legalrag.core.corpus import load_corpus
from legalrag.core.pipeline import AnswerBundle, Pipeline

DEFAULT_EXPERIMENT = "e0_naive_baseline"


# --------------------------------------------------------------------------
# Response models — the shared contract for every front door.
# --------------------------------------------------------------------------


class Citation(BaseModel):
    """A verified source behind an answer. Every field is corpus-grounded: the
    deterministic verifier has already dropped any citation whose quote/paragraph
    is not present in the source document."""

    doc_id: str
    title: str
    pinpoint: str  # human-facing, e.g. "Griswold v. Connecticut ¶ 12"
    para_ids: list[str] = Field(default_factory=list)
    quote: str | None = None
    source_url: str | None = None


class AnswerResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    verified: bool  # did the answer pass every verifier check
    verifier_flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    experiment: str
    index_id: str | None = None


class Passage(BaseModel):
    """A ranked retrieval hit — a source passage, no generation involved."""

    chunk_id: str
    doc_id: str
    title: str
    score: float
    rank: int
    pinpoint: str
    para_ids: list[str] = Field(default_factory=list)
    text: str = ""
    source_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    passages: list[Passage] = Field(default_factory=list)
    experiment: str
    index_id: str | None = None


class CaseNode(BaseModel):
    """A document's position in the citation graph: authority + treatment."""

    doc_id: str
    title: str
    authority: float
    status: str  # "good law" | "questioned" | "overruled"
    incoming_treatments: list[str] = Field(default_factory=list)
    cites: list[str] = Field(default_factory=list)
    cited_by: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------


class LegalRAGService:
    """Loads one experiment's pipeline and serves it. Construct via
    :meth:`load`, which mirrors the CLI's pipeline build."""

    def __init__(self, pipeline: Pipeline, experiment: str) -> None:
        self.pipeline = pipeline
        self.experiment = experiment
        self._lock = threading.Lock()

    @classmethod
    def load(cls, experiment: str = DEFAULT_EXPERIMENT) -> LegalRAGService:
        """Build and index the pipeline for ``experiment`` (does the expensive
        work once, at startup)."""
        cfg = load_experiment(experiment)
        corpus = load_corpus(cfg.corpus)
        pipeline = Pipeline(cfg, corpus)
        pipeline.build_index()
        return cls(pipeline, experiment)

    # --- helpers -------------------------------------------------------

    def _doc_title(self, doc_id: str) -> str:
        doc = self.pipeline.corpus.get(doc_id)
        return doc.title if doc else doc_id

    def _doc_url(self, doc_id: str) -> str | None:
        doc = self.pipeline.corpus.get(doc_id)
        return doc.source.url if doc else None

    def _pinpoint(self, doc_id: str, para_ids: list[str]) -> str:
        doc = self.pipeline.corpus.get(doc_id)
        if doc and para_ids:
            return doc.pinpoint(para_ids[0])
        return doc_id

    # --- operations ----------------------------------------------------

    def answer(self, query: str) -> AnswerResponse:
        """Grounded, citation-verified answer to ``query``."""
        with self._lock:
            bundle: AnswerBundle = self.pipeline.answer(query)
        ans = bundle.answer
        citations = [
            Citation(
                doc_id=c.doc_id,
                title=self._doc_title(c.doc_id),
                pinpoint=self._pinpoint(c.doc_id, c.para_ids),
                para_ids=c.para_ids,
                quote=c.quote,
                source_url=self._doc_url(c.doc_id),
            )
            for c in ans.citations
        ]
        flags = [
            f"{chk.name}: {', '.join(chk.offending)}" if chk.offending else chk.name
            for chk in bundle.report.failures()
        ]
        return AnswerResponse(
            query=query,
            answer=ans.text,
            abstained=ans.abstained,
            verified=bundle.report.passed,
            verifier_flags=flags,
            citations=citations,
            experiment=self.experiment,
            index_id=self.pipeline.index_id,
        )

    def search(self, query: str, k: int = 10) -> SearchResponse:
        """Ranked source passages for ``query`` (retrieval only, no generation)."""
        with self._lock:
            result = self.pipeline.retrieve(query)
        passages = [
            Passage(
                chunk_id=sc.chunk_id,
                doc_id=sc.doc_id,
                title=self._doc_title(sc.doc_id),
                score=sc.score,
                rank=sc.rank if sc.rank is not None else i,
                pinpoint=self._pinpoint(sc.doc_id, sc.para_ids),
                para_ids=sc.para_ids,
                text=sc.text,
                source_url=self._doc_url(sc.doc_id),
            )
            for i, sc in enumerate(result.scored[:k])
        ]
        return SearchResponse(
            query=query,
            passages=passages,
            experiment=self.experiment,
            index_id=self.pipeline.index_id,
        )

    def case(self, doc_id: str) -> CaseNode | None:
        """A case's citation-graph position, or ``None`` if not in the corpus."""
        if doc_id not in self.pipeline.corpus:
            return None
        g = self.pipeline.graph
        status = (
            "overruled"
            if g.is_overruled(doc_id)
            else ("questioned" if g.is_questioned(doc_id) else "good law")
        )
        return CaseNode(
            doc_id=doc_id,
            title=self._doc_title(doc_id),
            authority=g.authority_scores().get(doc_id, 0.0),
            status=status,
            incoming_treatments=[t.value for t in g.incoming_treatments(doc_id)],
            cites=list(g.cites(doc_id)),
            cited_by=list(g.cited_by(doc_id)),
        )
