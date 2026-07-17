"""US jurisdiction pack: regex citation extraction -> resolution -> graph.

Hermetic: no network, no eyecite. Exercises the pack end-to-end on the
hand-crafted US sample opinions.
"""

from __future__ import annotations

import pytest

from legalrag.core.citations import CitationResolver
from legalrag.core.corpus import Corpus
from legalrag.core.graph.citation_graph import CitationGraph
from legalrag.core.interfaces import Stage
from legalrag.core.models import CitationKind, Treatment
from legalrag.core.registry import REGISTRY
from legalrag.jurisdictions.us import citations as us_citations  # noqa: F401  (registers)
from legalrag.jurisdictions.us.citations import USRegexCitationExtractor, detect_treatment
from legalrag.jurisdictions.us.courts import court_for_reporter
from legalrag.jurisdictions.us.sample import build_us_sample

ROE = "us/scotus/410-us-113/v1"
CASEY = "us/scotus/505-us-833/v1"
DOBBS = "us/scotus/597-us-215/v1"
GRISWOLD = "us/scotus/381-us-479/v1"
ISAACSON = "us/scotus/716-f3d-1213/v1"


@pytest.fixture(scope="module")
def resolved_graph() -> CitationGraph:
    corpus = Corpus(build_us_sample())
    extractor = USRegexCitationExtractor()
    for doc in corpus:
        extractor.extract(doc)
    CitationResolver(corpus).resolve_corpus(corpus)
    return CitationGraph.from_corpus(corpus)


def test_pack_registers_extractors() -> None:
    assert "us_regex" in REGISTRY.available(Stage.CITATION_EXTRACTOR)
    assert "eyecite" in REGISTRY.available(Stage.CITATION_EXTRACTOR)


def test_reporter_extraction_and_normalization() -> None:
    ext = USRegexCitationExtractor(case_names=False)
    doc = next(d for d in build_us_sample() if d.doc_id == CASEY)
    ext.extract(doc)
    normals = {m.normalized for m in doc.citations_out}
    assert "410 U.S. 113" in normals            # Roe, normalized from "410 U.S. 113"
    assert all(m.kind == CitationKind.REPORTER for m in doc.citations_out)


def test_does_not_misread_f_supp_as_f() -> None:
    ext = USRegexCitationExtractor(case_names=False)
    from legalrag.core.models import (
        DocType,
        LegalDocument,
        Paragraph,
        Provenance,
        Section,
    )

    doc = LegalDocument(
        doc_id="us/x/1", doc_type=DocType.JUDGMENT, jurisdiction="us", title="X v. Y",
        source=Provenance(source="t"),
        body=[Section(paragraphs=[Paragraph(text="See 410 F. Supp. 3d 500 (D. Ariz. 2019).")])],
    )
    ext.extract(doc)
    normals = {m.normalized for m in doc.citations_out}
    assert "410 F. Supp. 3d 500" in normals
    assert "410 F. 500" not in normals          # overlap guard prevents mis-read


def test_treatment_heuristic() -> None:
    assert detect_treatment("These decisions are hereby overruled.", 6) == Treatment.OVERRULED
    assert detect_treatment("We reaffirm the holding.", 3) == Treatment.FOLLOWED
    assert detect_treatment("The court distinguished the prior case.", 10) == Treatment.DISTINGUISHED
    assert detect_treatment("The statute was enacted in 1970.", 4) is None


def test_resolution_builds_expected_edges(resolved_graph: CitationGraph) -> None:
    g = resolved_graph
    assert ROE in g.cites(CASEY)        # Casey cites Roe
    assert ROE in g.cites(DOBBS)        # Dobbs cites Roe
    assert CASEY in g.cites(DOBBS)      # Dobbs cites Casey
    assert GRISWOLD in g.cites(ROE)     # Roe cites Griswold
    assert ROE in g.cites(ISAACSON)     # circuit case applies Roe
    assert ISAACSON in g.cited_by(ROE)  # ...so Roe is cited by the circuit case


def test_overruled_treatment_flows_to_graph(resolved_graph: CitationGraph) -> None:
    # Dobbs overrules Roe and Casey -> both are no longer good law
    assert resolved_graph.is_overruled(ROE)
    assert resolved_graph.is_overruled(CASEY)
    assert not resolved_graph.is_overruled(GRISWOLD)


def test_authority_apex_above_appellate(resolved_graph: CitationGraph) -> None:
    scores = resolved_graph.authority_scores()
    # A current, good-law apex case outranks an appellate leaf. (Roe itself sits
    # low here — correctly — because it is overruled by Dobbs in this corpus.)
    assert scores[DOBBS] > scores[ISAACSON]
    # and the overruled apex case ranks below the apex case that overruled it
    assert scores[ROE] < scores[DOBBS]


def test_court_assigned_from_reporter() -> None:
    from legalrag.core.models import CourtLevel

    assert court_for_reporter("U.S.").level == CourtLevel.APEX          # type: ignore[union-attr]
    assert court_for_reporter("F.3d").level == CourtLevel.APPELLATE     # type: ignore[union-attr]
    assert court_for_reporter("F. Supp. 2d").level == CourtLevel.TRIAL  # type: ignore[union-attr]
    assert court_for_reporter("Nonsense") is None
