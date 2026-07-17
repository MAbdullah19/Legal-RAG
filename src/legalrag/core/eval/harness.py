"""Evaluation harness (master plan §6).

Runs a built pipeline over an eval set and computes retrieval + generation
metrics, with the per-query-type breakdown that legal IR needs (citation-lookup
vs conceptual vs multi-hop trade off differently). The headline safety metric —
**hallucinated-citation rate** — is measured on the post-verification answer and
must be 0 (RQ3).

Statistical significance (paired bootstrap, master plan §6.3) consumes the
per-query records this harness stores; ``legalrag compare`` uses them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from legalrag.core.eval import metrics as M
from legalrag.core.models import EvalQuery
from legalrag.core.pipeline import Pipeline


@dataclass(slots=True)
class QueryRecord:
    qid: str
    query_type: str
    answerable: bool
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: float = float("nan")
    rr: float = 0.0
    citation_precision: float = float("nan")
    citation_recall: float = float("nan")
    citation_f1: float = float("nan")
    hallucinated: int = 0
    abstained: bool = False
    verifier_passed: bool = True


@dataclass(slots=True)
class EvalReport:
    experiment: str
    n_queries: int
    metrics: dict[str, float]
    by_type: dict[str, dict[str, float]]
    records: list[QueryRecord] = field(default_factory=list)

    def flat_metrics(self) -> dict[str, float]:
        return dict(self.metrics)


def _doc_level(keys: set[str]) -> set[str]:
    return {k.split("#", 1)[0] for k in keys}


def evaluate(pipeline: Pipeline, queries: list[EvalQuery], ks: list[int], ndcg_k: int) -> EvalReport:
    records: list[QueryRecord] = []

    for q in queries:
        bundle = pipeline.answer(q.text, qid=q.qid)
        ranking = bundle.retrieval.doc_ranking()
        rel_grades = q.relevance()
        rel_set = set(rel_grades)

        rec = QueryRecord(qid=q.qid, query_type=str(q.query_type), answerable=q.answerable)
        if rel_set:
            rec.recall = {k: M.recall_at_k(ranking, rel_set, k) for k in ks}
            rec.ndcg = M.ndcg_at_k(ranking, rel_grades, ndcg_k)
            rec.rr = M.reciprocal_rank(ranking, rel_set)

        pred_docs = {c.doc_id for c in bundle.answer.citations}
        gold_docs = _doc_level(set(q.gold_citations)) or set(q.gold_doc_ids)
        if pred_docs or gold_docs:
            p, r, f1 = M.citation_prf(pred_docs, gold_docs)
            rec.citation_precision, rec.citation_recall, rec.citation_f1 = p, r, f1
        rec.hallucinated = sum(1 for d in pred_docs if d not in pipeline.corpus)
        rec.abstained = bundle.answer.abstained
        rec.verifier_passed = bundle.report.passed
        records.append(rec)

    report_metrics = _aggregate(records, ks)
    by_type = _aggregate_by_type(records, ks)
    return EvalReport(
        experiment=pipeline.config.name,
        n_queries=len(queries),
        metrics=report_metrics,
        by_type=by_type,
        records=records,
    )


def _aggregate(records: list[QueryRecord], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = M.mean([r.recall.get(k, float("nan")) for r in records])
    out["ndcg@10"] = M.mean([r.ndcg for r in records])
    out["mrr"] = M.mean([r.rr for r in records if r.recall])
    out["citation_precision"] = M.mean([r.citation_precision for r in records])
    out["citation_recall"] = M.mean([r.citation_recall for r in records])
    out["citation_f1"] = M.mean([r.citation_f1 for r in records])

    total_pred_hallucinations = sum(r.hallucinated for r in records)
    out["hallucinated_citation_rate"] = float(total_pred_hallucinations) / max(len(records), 1)

    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    out["answer_rate"] = (
        sum(1 for r in answerable if not r.abstained) / len(answerable)
        if answerable
        else float("nan")
    )
    out["over_abstention_rate"] = (
        sum(1 for r in answerable if r.abstained) / len(answerable)
        if answerable
        else float("nan")
    )
    out["correct_abstention_rate"] = (
        sum(1 for r in unanswerable if r.abstained) / len(unanswerable)
        if unanswerable
        else float("nan")
    )
    return out


def _aggregate_by_type(records: list[QueryRecord], ks: list[int]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in records:
        groups[r.query_type].append(r)
    return {qt: _aggregate(recs, ks) for qt, recs in sorted(groups.items())}


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "experiment": report.experiment,
        "n_queries": report.n_queries,
        "metrics": report.metrics,
        "by_type": report.by_type,
    }
