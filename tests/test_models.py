"""Contract tests for the canonical document model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from legalrag.core.models import (
    Citation,
    CitationKind,
    CourtLevel,
    CourtRef,
    DocType,
    EvalQuery,
    LegalDocument,
    Paragraph,
    Provenance,
    Qrel,
    RetrievalResult,
    ScoredChunk,
    Section,
    Treatment,
)
from legalrag.core.models.enums import NEGATIVE_TREATMENTS


def _doc(**overrides: object) -> LegalDocument:
    base: dict[str, object] = {
        "doc_id": "pk/sc/2018-595/v1",
        "doc_type": DocType.JUDGMENT,
        "jurisdiction": "pk",
        "title": "Alpha v. Beta",
        "source": Provenance(source="test"),
        "body": [
            Section(heading="Facts", path=["Facts"], paragraphs=[
                Paragraph(text="The appellant filed a petition."),
                Paragraph(para_no="2", text="The respondent objected on limitation."),
            ]),
            Section(heading="Held", path=["Held"], paragraphs=[
                Paragraph(text="The appeal is allowed."),
            ]),
        ],
    }
    base.update(overrides)
    return LegalDocument(**base)  # type: ignore[arg-type]


def test_pids_are_assigned_sequentially_and_stable() -> None:
    doc = _doc()
    pids = [p.pid for p in doc.paragraphs()]
    assert pids == ["p0001", "p0002", "p0003"]
    # explicit pid is preserved
    assert doc.get_paragraph("p0002").para_no == "2"  # type: ignore[union-attr]


def test_duplicate_explicit_pid_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc(body=[
            Section(paragraphs=[
                Paragraph(pid="pX", text="a"),
                Paragraph(pid="pX", text="b"),
            ])
        ])


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Provenance(source="x", nonsense="boom")  # type: ignore[call-arg]


def test_full_text_and_pinpoint() -> None:
    doc = _doc()
    assert "appeal is allowed" in doc.full_text()
    assert doc.pinpoint("p0002") == "Alpha v. Beta ¶ 2"
    # pid falls back to itself when no human number is present
    assert doc.pinpoint("p0001") == "Alpha v. Beta ¶ p0001"


def test_authority_weight_from_court_level() -> None:
    apex = _doc(court=CourtRef(court_id="pk/sc", name="SCP", level=CourtLevel.APEX, jurisdiction="pk"))
    trial = _doc(court=CourtRef(court_id="pk/dc", name="DC", level=CourtLevel.TRIAL, jurisdiction="pk"))
    assert apex.authority_weight() > trial.authority_weight()
    # missing court -> lowest default, never crashes
    assert _doc().authority_weight() == CourtLevel.OTHER.authority_weight


def test_citation_key_normalizes() -> None:
    a = Citation(raw="PLD 2018 SC 595", kind=CitationKind.REPORTER, normalized="PLD 2018 SC 595")
    b = Citation(raw="pld 2018 sc 595")
    assert a.key() == b.key()


def test_negative_treatments_membership() -> None:
    assert Treatment.OVERRULED in NEGATIVE_TREATMENTS
    assert Treatment.FOLLOWED not in NEGATIVE_TREATMENTS


def test_eval_query_relevance_merges_gold_and_qrels() -> None:
    q = EvalQuery(
        qid="q1",
        text="limitation in appeals",
        gold_doc_ids=["d1", "d2"],
        qrels=[Qrel(doc_id="d1", grade=3)],
    )
    rel = q.relevance()
    assert rel == {"d1": 3, "d2": 1}


def test_retrieval_result_doc_ranking_dedups_keeping_best() -> None:
    rr = RetrievalResult(qid="q1", scored=[
        ScoredChunk(chunk_id="d1#a", doc_id="d1", score=0.9),
        ScoredChunk(chunk_id="d1#b", doc_id="d1", score=0.5),
        ScoredChunk(chunk_id="d2#a", doc_id="d2", score=0.4),
    ])
    assert rr.doc_ranking() == ["d1", "d2"]
