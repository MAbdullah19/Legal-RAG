"""CourtListener connector — mapping + slice ingestion, fully hermetic.

The network is replaced by canned sources, so these test the *mapping and citation
wiring* (CL JSON -> canonical doc -> extracted + graph-resolved citations), never
the network. Covers both operating modes: full-text (token) and anonymous
(metadata + snippet + CL 'cites' graph edges).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from legalrag.core.corpus import read_documents
from legalrag.core.models import CourtLevel
from legalrag.jurisdictions.us.courtlistener import ingest_slice, to_document

_ROE: dict[str, Any] = {
    "cluster_id": 108713,
    "caseName": "Roe v. Wade",
    "court_id": "scotus",
    "court": "Supreme Court of the United States",
    "dateFiled": "1973-01-22",
    "citation": ["410 U.S. 113"],
    "opinions": [{"id": 1001, "cites": [], "snippet":
                  "This case presents the question of a constitutional right to privacy."}],
}
_CASEY: dict[str, Any] = {
    "cluster_id": 112786,
    "caseName": "Planned Parenthood v. Casey",
    "court_id": "scotus",
    "court": "Supreme Court of the United States",
    "dateFiled": "1992-06-29",
    "citation": ["505 U.S. 833"],
    "opinions": [{"id": 1002, "cites": [1001], "snippet":
                  "We are asked to reconsider Roe v. Wade, 410 U.S. 113, and we adhere to it."}],
}
_DETAIL: dict[int, dict[str, Any]] = {
    1001: {"plain_text": (
        "This case presents the question of a constitutional right to privacy.\n\n"
        "We hold that this right is broad enough to encompass the decision at issue here."
    )},
    1002: {"plain_text": (
        "We are asked to reconsider Roe v. Wade, 410 U.S. 113, decided two decades ago.\n\n"
        "Adhering to Roe, the Court reaffirms its essential holding under a new standard."
    )},
}


class _FakeSource:
    def __init__(self, has_token: bool) -> None:
        self.has_token = has_token
        self.detail_calls: list[int | str] = []

    def search_opinions(
        self,
        query: str | None = None,
        court: str | None = None,
        filed_after: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return [_ROE, _CASEY][:limit]

    def opinion_detail(self, opinion_id: int | str) -> dict[str, Any]:
        self.detail_calls.append(opinion_id)
        return _DETAIL[int(opinion_id)]


def test_to_document_maps_core_fields() -> None:
    doc = to_document(_ROE, _DETAIL[1001])
    assert doc is not None
    assert doc.doc_id == "us/scotus/108713/v1"
    assert doc.title == "Roe v. Wade"
    assert doc.court is not None and doc.court.level is CourtLevel.APEX
    assert doc.citations_self[0].normalized == "410 U.S. 113"
    assert len(list(doc.iter_paragraphs())) == 2  # full text: two blank-line paragraphs


def test_to_document_snippet_fallback_without_detail() -> None:
    doc = to_document(_ROE, None)  # anonymous: no detail -> use the search snippet
    assert doc is not None
    assert len(list(doc.iter_paragraphs())) == 1
    assert "right to privacy" in doc.full_text()


def test_to_document_skips_empty_body() -> None:
    assert to_document({"cluster_id": 1}, {"plain_text": ""}) is None


def test_ingest_full_text_mode(tmp_path: Path) -> None:
    src = _FakeSource(has_token=True)
    report = ingest_slice(src, slice_name="unit", limit=10, out_dir=tmp_path)

    assert report.full_text is True
    assert src.detail_calls == [1001, 1002]      # one detail fetch per lead opinion
    assert report.n_docs == 2
    assert report.by_court_level == {"apex": 2}
    assert report.n_graph_edges >= 1             # Casey cites Roe's opinion id 1001
    assert report.n_resolved >= 1 and report.resolution_rate > 0.0

    written = read_documents(tmp_path / "documents.jsonl")
    casey = next(d for d in written if d.doc_id == "us/scotus/112786/v1")
    assert any(m.target_doc_id == "us/scotus/108713/v1" for m in casey.citations_out)


def test_ingest_anonymous_mode_builds_graph_edges(tmp_path: Path) -> None:
    src = _FakeSource(has_token=False)
    report = ingest_slice(src, slice_name="unit_anon", limit=10, out_dir=tmp_path)

    assert report.full_text is False
    assert src.detail_calls == []                # no detail fetches without a token
    assert report.n_docs == 2
    assert report.n_graph_edges == 1             # exact edge from CL 'cites', no full text
    casey = next(
        d for d in read_documents(tmp_path / "documents.jsonl")
        if d.doc_id == "us/scotus/112786/v1"
    )
    assert any(m.target_doc_id == "us/scotus/108713/v1" for m in casey.citations_out)
