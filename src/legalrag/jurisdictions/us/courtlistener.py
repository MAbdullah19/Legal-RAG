"""CourtListener ingestion connector (US annex §1; master plan §5.1, P1).

Pulls real US opinions from the Free Law Project's CourtListener v4 REST API and
maps them into the canonical :class:`LegalDocument`, so the ablation ladder runs on
real case law instead of the synthetic fixture. Deliberately dependency-light
(stdlib ``urllib``) and network-isolated from everything else: the data source is
injectable, so tests mock it and the rest of the engine stays hermetic (ADR 0003).

**Graceful degradation** around CourtListener's auth model:

- *Anonymous* (no token): the ``/search/`` endpoint returns rich metadata + each
  opinion's ``cites`` (the opinion ids it cites) + a text snippet. That is enough to
  build a real **citation-graph** slice — authority, treatment, precedent chains —
  the most novel component (E7), on real data, with no key.
- *With a token* (``COURTLISTENER_API_TOKEN``): the connector also fetches full
  opinion bodies from ``/opinions/{id}/``, upgrading the same slice to full-text
  retrieval (E1/E5). Get a free token at courtlistener.com/help/api.

Public-record data only; every doc records its CourtListener URL + license basis
(governance §12).

Run:  ``python -m legalrag.jurisdictions.us.courtlistener --slice clerc-dev \\
          --query "right to privacy" --limit 30``
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from legalrag.core.citations import CitationResolver
from legalrag.core.corpus import Corpus, corpus_dir, write_documents
from legalrag.core.models import (
    Citation,
    CitationKind,
    CitationMention,
    DocType,
    LegalDocument,
    LicenseBasis,
    Paragraph,
    Provenance,
    Section,
)
from legalrag.jurisdictions.us.citations import USRegexCitationExtractor
from legalrag.jurisdictions.us.courts import court_ref_for_courtlistener_id

_API = "https://www.courtlistener.com/api/rest/v4"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


class _Source(Protocol):
    """What ``ingest_slice`` needs from a data source — satisfied by
    CourtListenerClient and by a mock in tests (keeps ingestion hermetic)."""

    @property
    def has_token(self) -> bool: ...

    def search_opinions(
        self,
        query: str | None = ...,
        court: str | None = ...,
        filed_after: str | None = ...,
        limit: int = ...,
    ) -> list[dict[str, Any]]: ...

    def opinion_detail(self, opinion_id: int | str) -> dict[str, Any]: ...


class CourtListenerClient:
    """Thin, polite wrapper over the CourtListener v4 API (stdlib only)."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = _API,
        delay: float = 0.5,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("COURTLISTENER_API_TOKEN")
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    def get_json(self, url: str) -> dict[str, Any]:
        headers = {"User-Agent": "legalrag-research/0.1 (academic; contact via repo)"}
        if self.token:
            headers["Authorization"] = f"Token {self.token}"
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    time.sleep(self.delay)  # be polite between calls
                    data: dict[str, Any] = json.load(resp)
                    return data
            except urllib.error.HTTPError as e:
                # 429 (throttled) and anonymous 401 (transient rate block) are retryable.
                if e.code in (401, 429) and attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")

    def search_opinions(
        self,
        query: str | None = None,
        court: str | None = None,
        filed_after: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {"type": "o", "order_by": "score desc"}
        if query:
            params["q"] = query
        if court:
            params["court"] = court
        if filed_after:
            params["filed_after"] = filed_after
        results: list[dict[str, Any]] = []
        url = f"{self.base_url}/search/?{urllib.parse.urlencode(params)}"
        while url and len(results) < limit:
            page = self.get_json(url)
            results.extend(page.get("results", []))
            url = page.get("next") or ""
        return results[:limit]

    def opinion_detail(self, opinion_id: int | str) -> dict[str, Any]:
        return self.get_json(f"{self.base_url}/opinions/{opinion_id}/")


# --------------------------------------------------------------------------
# Mapping: CourtListener JSON -> canonical LegalDocument
# --------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    text = _TAG_RE.sub("", raw or "")  # strip any HTML
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _WS_RE.sub(" ", text)


def _paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """Split opinion text into paragraph-sized units on blank lines, dropping
    boilerplate-short fragments. Paragraph is the atomic pinpoint unit (§4)."""
    blocks = re.split(r"\n\s*\n", text)
    paras = [" ".join(b.split()) for b in blocks]
    return [p for p in paras if len(p) >= min_chars]


def _snippet_text(meta: dict[str, Any]) -> str:
    parts = [
        _clean_text(o.get("snippet") or "")
        for o in (meta.get("opinions") or [])
        if isinstance(o, dict) and o.get("snippet")
    ]
    return "\n\n".join(p for p in parts if p)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def to_document(meta: dict[str, Any], detail: dict[str, Any] | None = None) -> LegalDocument | None:
    """Build a canonical doc from a search result (``meta``), using full text from
    ``detail`` when available, else the search snippet. Returns None if there is no
    usable body at all (quarantine, not crash — §5.1)."""
    cluster_id = meta.get("cluster_id") or meta.get("id")
    if cluster_id is None:
        return None

    body_raw = ""
    if detail:
        body_raw = detail.get("plain_text") or detail.get("html_with_citations") or ""
    body_text = _clean_text(body_raw) if body_raw else _snippet_text(meta)
    paras = _paragraphs(body_text)
    if not paras:  # fall back to editorial fields so the doc is still graph-usable
        fallback = _clean_text(meta.get("syllabus") or meta.get("procedural_history") or "")
        paras = _paragraphs(fallback, min_chars=20)
    if not paras:
        return None

    court_id = str(meta.get("court_id") or "unknown")
    citations = [c for c in (meta.get("citation") or []) if isinstance(c, str) and c.strip()]
    return LegalDocument(
        doc_id=f"us/{court_id}/{cluster_id}/v1",
        doc_type=DocType.JUDGMENT,
        jurisdiction="us",
        title=str(meta.get("caseName") or f"cluster {cluster_id}"),
        court=court_ref_for_courtlistener_id(court_id, meta.get("court")),
        decision_date=_parse_date(meta.get("dateFiled")),
        citations_self=[
            Citation(raw=c, kind=CitationKind.REPORTER, normalized=" ".join(c.split()))
            for c in citations
        ],
        source=Provenance(
            source="us/courtlistener",
            url=f"https://www.courtlistener.com/opinion/{cluster_id}/",
            license_basis=LicenseBasis.PUBLIC_RECORD,
            license_note="US judicial opinion (public record) via CourtListener/Free Law Project",
        ),
        body=[Section(heading="Opinion", path=["Opinion"],
                      paragraphs=[Paragraph(text=t) for t in paras])],
    )


def _opinion_ids(meta: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for op in meta.get("opinions") or []:
        if isinstance(op, dict) and op.get("id") is not None:
            with contextlib.suppress(TypeError, ValueError):
                ids.append(int(op["id"]))
    return ids


def _cited_opinion_ids(meta: dict[str, Any]) -> set[int]:
    cited: set[int] = set()
    for op in meta.get("opinions") or []:
        for c in (op.get("cites") or []) if isinstance(op, dict) else []:
            with contextlib.suppress(TypeError, ValueError):
                cited.add(int(c))
    return cited


# --------------------------------------------------------------------------
# Slice ingestion + quality report
# --------------------------------------------------------------------------

@dataclass(slots=True)
class IngestReport:
    slice_name: str
    n_docs: int = 0
    n_skipped: int = 0
    n_mentions: int = 0
    n_resolved: int = 0
    n_graph_edges: int = 0          # exact edges from CourtListener 'cites'
    full_text: bool = False
    by_court_level: dict[str, int] = field(default_factory=dict)
    out_path: str = ""

    @property
    def resolution_rate(self) -> float:
        return self.n_resolved / self.n_mentions if self.n_mentions else 0.0


def ingest_slice(
    source: _Source,
    *,
    slice_name: str,
    query: str | None = None,
    court: str | None = None,
    filed_after: str | None = None,
    limit: int = 20,
    out_dir: Path | None = None,
) -> IngestReport:
    """Pull a slice, extract + resolve citations (text + CL graph edges), write
    JSONL, and return a quality report."""
    search = source.search_opinions(query=query, court=court, filed_after=filed_after, limit=limit)
    use_full_text = bool(getattr(source, "has_token", False))

    extractor = USRegexCitationExtractor()
    report = IngestReport(slice_name=slice_name, full_text=use_full_text)
    docs: list[LegalDocument] = []
    op_to_doc: dict[int, str] = {}
    cited_by_doc: dict[str, set[int]] = {}

    for meta in search:
        detail = None
        if use_full_text:
            op_ids = _opinion_ids(meta)
            if op_ids:
                detail = source.opinion_detail(op_ids[0])
        doc = to_document(meta, detail)
        if doc is None:
            report.n_skipped += 1
            continue
        extractor.extract(doc)  # text-derived mentions (from full text or snippet)
        docs.append(doc)
        for oid in _opinion_ids(meta):
            op_to_doc[oid] = doc.doc_id
        cited_by_doc[doc.doc_id] = _cited_opinion_ids(meta)
        if doc.court is not None:
            level = doc.court.level.value
            report.by_court_level[level] = report.by_court_level.get(level, 0) + 1

    # Exact citation edges from CourtListener's own resolution (works anonymously).
    doc_by_id = {d.doc_id: d for d in docs}
    for doc in docs:
        existing = {m.target_doc_id for m in doc.citations_out if m.target_doc_id}
        targets = {op_to_doc[c] for c in cited_by_doc.get(doc.doc_id, set()) if c in op_to_doc}
        targets.discard(doc.doc_id)
        for tgt in targets - existing:
            tdoc = doc_by_id[tgt]
            raw = tdoc.citations_self[0].raw if tdoc.citations_self else tdoc.title
            doc.citations_out.append(
                CitationMention(
                    raw=raw, kind=CitationKind.REPORTER, normalized=raw, target_doc_id=tgt
                )
            )
            report.n_graph_edges += 1

    corpus = Corpus(docs)
    resolver = CitationResolver(corpus)
    regex_resolved = resolver.resolve_corpus(corpus)
    report.n_docs = len(docs)
    report.n_mentions = sum(len(d.citations_out) for d in docs)
    report.n_resolved = report.n_graph_edges + regex_resolved

    target = (out_dir or corpus_dir(f"us/{slice_name}")) / "documents.jsonl"
    write_documents(docs, target)
    report.out_path = str(target)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest a US opinion slice from CourtListener.")
    ap.add_argument("--slice", default="clerc-dev", help="slice name -> evalsets/us/<slice>/")
    ap.add_argument("--query", default=None, help="full-text search query")
    ap.add_argument("--court", default=None, help="court id filter, e.g. scotus, ca9")
    ap.add_argument("--filed-after", default=None, help="YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)

    client = CourtListenerClient()
    report = ingest_slice(
        client,
        slice_name=args.slice,
        query=args.query,
        court=args.court,
        filed_after=args.filed_after,
        limit=args.limit,
    )
    mode = "full-text (token)" if report.full_text else "metadata+snippet (anonymous)"
    print(f"slice={report.slice_name}  mode={mode}")
    print(f"docs={report.n_docs}  skipped={report.n_skipped}  court levels={report.by_court_level}")
    print(f"citation mentions={report.n_mentions}  graph edges(CL)={report.n_graph_edges}  "
          f"resolved(in-corpus)={report.n_resolved}  resolution_rate={report.resolution_rate:.3f}")
    print(f"wrote -> {report.out_path}")


if __name__ == "__main__":
    main()
