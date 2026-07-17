"""Unit tests for the metric functions (master plan §6.2)."""

from __future__ import annotations

import math

from legalrag.core.eval import metrics as M


def test_recall_at_k() -> None:
    ranking = ["d1", "d3", "d2", "d9"]
    rel = {"d1", "d2"}
    assert M.recall_at_k(ranking, rel, 1) == 0.5
    assert M.recall_at_k(ranking, rel, 3) == 1.0
    assert math.isnan(M.recall_at_k(ranking, set(), 3))


def test_reciprocal_rank() -> None:
    assert M.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert M.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_prefers_correct_order() -> None:
    grades = {"a": 3, "b": 2, "c": 0}
    good = M.ndcg_at_k(["a", "b", "c"], grades, 3)
    bad = M.ndcg_at_k(["c", "b", "a"], grades, 3)
    assert good == 1.0
    assert good > bad


def test_citation_prf() -> None:
    p, r, f1 = M.citation_prf({"a", "b"}, {"a"})
    assert p == 0.5 and r == 1.0
    assert math.isclose(f1, 2 * 0.5 * 1.0 / 1.5)
    # both empty -> perfect (nothing claimed, nothing required)
    assert M.citation_prf(set(), set()) == (1.0, 1.0, 1.0)


def test_mean_ignores_nan() -> None:
    assert M.mean([1.0, math.nan, 3.0]) == 2.0
    assert math.isnan(M.mean([math.nan]))
