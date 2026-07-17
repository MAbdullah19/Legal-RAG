"""Jurisdiction-neutral citation resolution (master plan §5.5)."""

from __future__ import annotations

from legalrag.core.citations import CitationResolver
from legalrag.core.corpus import Corpus
from legalrag.core.models import (
    Citation,
    CitationKind,
    CitationMention,
    DocType,
    LegalDocument,
    Paragraph,
    Provenance,
    Section,
)


def _doc(doc_id: str, title: str, self_cite: str, out: list[CitationMention]) -> LegalDocument:
    return LegalDocument(
        doc_id=doc_id,
        doc_type=DocType.JUDGMENT,
        jurisdiction="xx",
        title=title,
        citations_self=[Citation(raw=self_cite, kind=CitationKind.REPORTER, normalized=self_cite)],
        source=Provenance(source="test"),
        body=[Section(paragraphs=[Paragraph(text="body")])],
        citations_out=out,
    )


def _corpus() -> Corpus:
    # A cites B by reporter and a dangling cite; B cites A by case name.
    a = _doc(
        "xx/1", "Alpha v. Beta", "10 TLR 1",
        out=[
            CitationMention(raw="20 TLR 2", normalized="20 TLR 2"),
            CitationMention(raw="999 XX 9", normalized="999 XX 9"),  # dangling
        ],
    )
    b = _doc(
        "xx/2", "Gamma v. Delta", "20 TLR 2",
        out=[CitationMention(raw="Alpha v. Beta")],  # case-name reference
    )
    return Corpus([a, b])


def test_resolves_reporter_and_case_name() -> None:
    corpus = _corpus()
    r = CitationResolver(corpus)
    n = r.resolve_corpus(corpus)
    assert n == 2  # A->B (reporter) and B->A (case name); dangling stays unresolved
    a = corpus.get("xx/1")
    b = corpus.get("xx/2")
    assert a is not None and b is not None
    assert a.citations_out[0].target_doc_id == "xx/2"
    assert a.citations_out[1].target_doc_id is None  # dangling, never fabricated
    assert b.citations_out[0].target_doc_id == "xx/1"


def test_resolution_rate() -> None:
    corpus = _corpus()
    r = CitationResolver(corpus)
    # 3 mentions total, 2 resolvable
    assert abs(r.resolution_rate(corpus) - 2 / 3) < 1e-9


def test_does_not_overwrite_existing_unless_asked() -> None:
    corpus = _corpus()
    a = corpus.get("xx/1")
    assert a is not None
    a.citations_out[0].target_doc_id = "manual/target"
    r = CitationResolver(corpus)
    r.resolve_corpus(corpus)
    assert a.citations_out[0].target_doc_id == "manual/target"  # preserved
    r.resolve_corpus(corpus, overwrite=True)
    assert a.citations_out[0].target_doc_id == "xx/2"  # now recomputed


def test_feeds_the_citation_graph() -> None:
    from legalrag.core.graph.citation_graph import CitationGraph

    corpus = _corpus()
    CitationResolver(corpus).resolve_corpus(corpus)
    g = CitationGraph.from_corpus(corpus)
    assert "xx/2" in g.cites("xx/1")
    assert "xx/1" in g.cites("xx/2")
