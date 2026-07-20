"""Qdrant adapter — pure mapping logic, hermetic (no server, no qdrant-client).

The Qdrant calls live inside methods behind a lazy import, so the payload/filter
translation is unit-testable on its own. Integration against a live Qdrant is
validated when the docker stack is up (docs/deployment.md)."""

from __future__ import annotations

from legalrag.core import builtins as _builtins  # noqa: F401  (registers components)
from legalrag.core.interfaces import Stage
from legalrag.core.models import Chunk
from legalrag.core.registry import REGISTRY
from legalrag.core.retrieve.vector_external import (
    _point_id,
    filter_conditions,
    payload_for,
)


def test_registered() -> None:
    assert "qdrant" in REGISTRY.available(Stage.VECTOR_INDEX)


def test_point_id_is_deterministic() -> None:
    assert _point_id("d#p1-p2") == _point_id("d#p1-p2")
    assert _point_id("a") != _point_id("b")


def test_payload_carries_ids_text_and_filter_fields() -> None:
    chunk = Chunk(
        chunk_id="us/scotus/1/v1#p0001",
        doc_id="us/scotus/1/v1",
        para_ids=["p0001"],
        text="the body",
        payload={"court_level": "apex", "jurisdiction": "us"},
    )
    p = payload_for(chunk)
    assert p["chunk_id"] == chunk.chunk_id
    assert p["doc_id"] == chunk.doc_id
    assert p["para_ids"] == ["p0001"]
    assert p["text"] == "the body"
    assert p["court_level"] == "apex" and p["jurisdiction"] == "us"


def test_filter_conditions_membership_vs_equality() -> None:
    assert filter_conditions(None) == []
    assert filter_conditions({}) == []
    conds = filter_conditions({"jurisdiction": "us", "court_level": ["apex", "appellate"]})
    assert ("jurisdiction", "eq", "us") in conds
    assert ("court_level", "any", ["apex", "appellate"]) in conds
