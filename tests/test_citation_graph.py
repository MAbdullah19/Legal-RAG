"""Citation graph + authority scoring (master plan §5.5)."""

from __future__ import annotations

import pytest

from legalrag.core.corpus import load_corpus
from legalrag.core.graph.citation_graph import CitationGraph


@pytest.fixture(scope="module")
def graph() -> CitationGraph:
    return CitationGraph.from_corpus(load_corpus("fixture"))


def test_edges_resolved_from_citations_out(graph: CitationGraph) -> None:
    # D3 (Rahim Trading) cites D4 (foundational) and D5 (Zephyr)
    assert "xx/apex/1990-003/v1" in graph.cites("xx/apex/2020-077/v1")
    assert "xx/high/2015-201/v1" in graph.cites("xx/apex/2020-077/v1")
    # D4 is cited by several
    assert "xx/apex/2020-077/v1" in graph.cited_by("xx/apex/1990-003/v1")


def test_overruled_detection(graph: CitationGraph) -> None:
    # D6 overrules D5 (Zephyr) -> Zephyr is no longer good law
    assert graph.is_overruled("xx/high/2015-201/v1")
    assert graph.is_questioned("xx/high/2015-201/v1")
    # the foundational case is followed, not questioned
    assert not graph.is_questioned("xx/apex/1990-003/v1")


def test_circular_citation_does_not_hang(graph: CitationGraph) -> None:
    # D7 <-> D8 cite each other; authority must still converge and be finite
    scores = graph.authority_scores()
    for d in ("xx/high/2017-330/v1", "xx/high/2017-331/v1"):
        assert 0.0 <= scores[d] <= 1.0


def test_authority_orders_apex_above_trial_noise(graph: CitationGraph) -> None:
    scores = graph.authority_scores()
    foundational = scores["xx/apex/1990-003/v1"]   # heavily-cited apex case
    trial_noise = scores["xx/trial/2019-905/v1"]   # off-topic trial order, uncited
    assert foundational > trial_noise


def test_overruled_case_is_penalized(graph: CitationGraph) -> None:
    scores = graph.authority_scores()
    # Zephyr (overruled) should score below the apex case that overruled it
    assert scores["xx/high/2015-201/v1"] < scores["xx/apex/2021-090/v1"]


def test_neighbours_sorted_by_authority(graph: CitationGraph) -> None:
    nbrs = graph.neighbours("xx/high/2015-201/v1")
    assert nbrs  # Zephyr connects to D3/D4/D6
    auth = graph.authority_scores()
    scores = [auth[n] for n in nbrs]
    assert scores == sorted(scores, reverse=True)
