"""The deterministic verifier is the anti-hallucination guard (RQ3). These tests
pin its core promise: a citation not provably in the corpus is never emitted."""

from __future__ import annotations

from legalrag.core.corpus import Corpus
from legalrag.core.models import (
    AnswerCitation,
    DocType,
    GeneratedAnswer,
    LegalDocument,
    Paragraph,
    Provenance,
    Section,
)
from legalrag.core.verify.checks import DeterministicVerifier


def _corpus() -> Corpus:
    doc = LegalDocument(
        doc_id="xx/apex/1/v1",
        doc_type=DocType.JUDGMENT,
        jurisdiction="xx",
        title="Real v. Case",
        source=Provenance(source="test"),
        body=[Section(paragraphs=[Paragraph(text="The rule of remoteness limits recoverable damages.")])],
    )
    return Corpus([doc])


def test_valid_citation_passes() -> None:
    v = DeterministicVerifier(_corpus())
    ans = GeneratedAnswer(
        qid="q",
        text="...",
        citations=[AnswerCitation(doc_id="xx/apex/1/v1", para_ids=["p0001"],
                                  quote="The rule of remoteness limits recoverable damages.")],
    )
    out, report = v.verify(ans)
    assert report.passed
    assert len(out.citations) == 1


def test_fabricated_document_is_stripped() -> None:
    v = DeterministicVerifier(_corpus())
    ans = GeneratedAnswer(
        qid="q",
        text="...",
        citations=[AnswerCitation(doc_id="xx/apex/999/v1", para_ids=["p0001"], quote="made up")],
    )
    out, report = v.verify(ans)
    assert not report.passed
    assert out.citations == []           # fabricated citation never emitted
    assert out.abstained                 # nothing survived -> abstain


def test_nonexistent_paragraph_is_stripped() -> None:
    v = DeterministicVerifier(_corpus())
    ans = GeneratedAnswer(
        qid="q", text="...",
        citations=[AnswerCitation(doc_id="xx/apex/1/v1", para_ids=["p9999"])],
    )
    out, report = v.verify(ans)
    assert not report.passed
    assert out.citations == []


def test_misquote_is_flagged_and_dropped() -> None:
    v = DeterministicVerifier(_corpus())
    ans = GeneratedAnswer(
        qid="q", text="...",
        citations=[AnswerCitation(doc_id="xx/apex/1/v1", para_ids=["p0001"],
                                  quote="a sentence that never appears in the source")],
    )
    out, report = v.verify(ans)
    quote_check = next(c for c in report.checks if c.name == "quote_match")
    assert not quote_check.passed
    assert out.citations == []
