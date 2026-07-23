"""Neutral service core — end-to-end over the deterministic E0 pipeline.

Hermetic: E0 is fully in-process (hashing embedder, in-memory index, extractive
generator, deterministic verifier), so this needs no services, keys, fastapi, or
mcp. It asserts the served contract — grounded answers, verified citations,
ranked passages, graph lookup — the two adapters both depend on.
"""

from __future__ import annotations

import pytest

from legalrag.service.core import (
    AnswerResponse,
    LegalRAGService,
    SearchResponse,
)


@pytest.fixture(scope="module")
def service() -> LegalRAGService:
    return LegalRAGService.load("e0_naive_baseline")


def test_load_builds_index(service: LegalRAGService) -> None:
    assert service.experiment == "e0_naive_baseline"
    assert service.pipeline.index_id is not None
    assert len(service.pipeline.corpus) > 0


def test_answer_is_grounded_and_verified(service: LegalRAGService) -> None:
    resp = service.answer("damages for remoteness")
    assert isinstance(resp, AnswerResponse)
    assert resp.answer
    assert resp.verified is True  # E0 exit criterion: zero hallucinated citations
    assert resp.verifier_flags == []
    # Every returned citation is corpus-grounded and pinpoint-resolved.
    for c in resp.citations:
        assert c.doc_id in service.pipeline.corpus
        assert c.pinpoint and c.pinpoint != c.doc_id
    assert AnswerResponse.model_validate_json(resp.model_dump_json())


def test_search_returns_ranked_passages(service: LegalRAGService) -> None:
    resp = service.search("damages for remoteness", k=3)
    assert isinstance(resp, SearchResponse)
    assert 1 <= len(resp.passages) <= 3
    ranks = [p.rank for p in resp.passages]
    assert ranks == sorted(ranks)
    for p in resp.passages:
        assert p.doc_id in service.pipeline.corpus
        assert p.title


def test_search_respects_k(service: LegalRAGService) -> None:
    assert len(service.search("contract", k=1).passages) <= 1


def test_case_lookup_and_miss(service: LegalRAGService) -> None:
    known = next(iter(service.pipeline.corpus)).doc_id
    node = service.case(known)
    assert node is not None
    assert node.doc_id == known
    assert node.status in {"good law", "questioned", "overruled"}
    assert service.case("nope/not/a/doc") is None
