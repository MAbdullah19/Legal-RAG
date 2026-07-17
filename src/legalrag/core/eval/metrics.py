"""Retrieval and generation metrics (master plan §6.2). Pure functions over
rankings and citation sets so they are trivially unit-testable and independent
of any component."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return math.nan
    top = set(ranking[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranking[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[str], grades: dict[str, int], k: int) -> float:
    if not grades:
        return math.nan
    gains = [float(grades.get(d, 0)) for d in ranking[:k]]
    ideal = sorted((float(g) for g in grades.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def citation_prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    """Precision, recall, F1 over citation keys (compared at doc level)."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def mean(values: Sequence[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan
