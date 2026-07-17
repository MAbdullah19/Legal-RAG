"""Query-time and evaluation data structures.

These are the values that flow through the query pipeline (retrieval ->
generation -> verification) and into the eval harness. Keeping them as typed
models (not dicts) is what lets every stage be swapped without silent contract
drift.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from legalrag.core.models.enums import QueryType


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Queries and relevance judgments (evaluation inputs)
# --------------------------------------------------------------------------

class Qrel(_Model):
    """Graded relevance judgment for retrieval metrics."""

    doc_id: str
    grade: int = 1          # 0 = irrelevant, higher = more relevant


class EvalQuery(_Model):
    qid: str
    text: str
    query_type: QueryType = QueryType.CONCEPTUAL
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_citations: list[str] = Field(default_factory=list)  # "doc_id#pid" or "doc_id"
    qrels: list[Qrel] = Field(default_factory=list)
    answerable: bool = True
    as_of: date | None = None
    filters: dict[str, Any] = Field(default_factory=dict)

    def relevance(self) -> dict[str, int]:
        """doc_id -> grade, unifying explicit qrels and gold_doc_ids."""
        rel = {q.doc_id: q.grade for q in self.qrels}
        for doc_id in self.gold_doc_ids:
            rel.setdefault(doc_id, 1)
        return rel


# --------------------------------------------------------------------------
# Retrieval outputs
# --------------------------------------------------------------------------

class ScoredChunk(_Model):
    chunk_id: str
    doc_id: str
    score: float
    para_ids: list[str] = Field(default_factory=list)
    text: str = ""
    rank: int | None = None
    components: dict[str, float] = Field(default_factory=dict)  # e.g. dense/sparse subscores


class RetrievalResult(_Model):
    qid: str
    scored: list[ScoredChunk] = Field(default_factory=list)

    def doc_ranking(self) -> list[str]:
        """Unique doc_ids in rank order (first, best-scoring occurrence wins)."""
        seen: set[str] = set()
        order: list[str] = []
        for sc in self.scored:
            if sc.doc_id not in seen:
                seen.add(sc.doc_id)
                order.append(sc.doc_id)
        return order

    def top(self, k: int) -> list[ScoredChunk]:
        return self.scored[:k]


# --------------------------------------------------------------------------
# Generation + verification outputs
# --------------------------------------------------------------------------

class AnswerCitation(_Model):
    doc_id: str
    para_ids: list[str] = Field(default_factory=list)
    quote: str | None = None            # verbatim span, checked against source

    def pinpoint_key(self) -> str:
        head = self.para_ids[0] if self.para_ids else ""
        return f"{self.doc_id}#{head}" if head else self.doc_id


class GeneratedAnswer(_Model):
    qid: str
    text: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    abstained: bool = False


class CheckResult(_Model):
    name: str
    passed: bool
    detail: str = ""
    offending: list[str] = Field(default_factory=list)


class VerificationReport(_Model):
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# --------------------------------------------------------------------------
# Run bookkeeping (reproducibility, master plan §3.2 / §6.4)
# --------------------------------------------------------------------------

class RunManifest(_Model):
    experiment: str
    config_hash: str
    corpus_id: str
    index_id: str | None = None
    git_commit: str | None = None
    seed: int = 0
    model_versions: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str = ""
