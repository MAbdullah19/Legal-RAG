"""US citation extraction (US annex §2).

Two implementations behind the core ``CitationExtractor`` Protocol:

- ``us_regex`` (default): dependency-free reporter + case-name extraction with a
  rule-based treatment heuristic. Hermetic and fully tested — the same
  regex-grammar approach the Pakistan ``pk-cite`` extractor will take.
- ``eyecite`` (opt-in, lazy): the Free Law Project's mature extractor for full
  reporter coverage, pin cites, and id./supra resolution. ``pip install
  -e ".[citations]"``.

Both only *extract and normalize* mentions; linking a mention to a target
document is the jurisdiction-neutral :class:`~legalrag.core.citations.CitationResolver`.
Treatment here is a rule-based v0 (keyword heuristic); the plan's few-shot →
fine-tuned treatment classifier registers as a separate component later.
"""

from __future__ import annotations

import re

from legalrag.core.interfaces import Stage
from legalrag.core.models import CitationKind, CitationMention, LegalDocument, Treatment
from legalrag.core.registry import register

# Canonical reporter, regex body. Specific forms first so "F. Supp. 3d" is not
# mis-read as "F." — an overlap guard also enforces this.
_REPORTERS: list[tuple[str, str]] = [
    ("U.S.", r"U\.\s?S\."),
    ("S. Ct.", r"S\.\s?Ct\."),
    ("L. Ed. 2d", r"L\.\s?Ed\.\s?2d"),
    ("L. Ed.", r"L\.\s?Ed\."),
    ("F. Supp. 3d", r"F\.\s?Supp\.\s?3d"),
    ("F. Supp. 2d", r"F\.\s?Supp\.\s?2d"),
    ("F. Supp.", r"F\.\s?Supp\."),
    ("F.4th", r"F\.\s?4th"),
    ("F.3d", r"F\.\s?3d"),
    ("F.2d", r"F\.\s?2d"),
    ("F. App'x", r"F\.\s?App'x"),
    ("F.", r"F\."),
]
_REPORTER_RES: list[tuple[str, re.Pattern[str]]] = [
    (canon, re.compile(rf"(\d+)\s+{body}\s+(\d+)")) for canon, body in _REPORTERS
]

_CASE_NAME = re.compile(
    r"\b([A-Z][A-Za-z.'&-]+(?:\s+[A-Z][A-Za-z.'&-]+){0,4})\s+v\.\s+"
    r"([A-Z][A-Za-z.'&-]+(?:\s+[A-Za-z.'&-]+){0,4})"
)

# Rule-based treatment heuristic (v0). Ordered by strength.
_TREATMENT_KEYWORDS: list[tuple[Treatment, tuple[str, ...]]] = [
    (Treatment.OVERRULED, ("overrul", "abrogat")),
    (Treatment.DISTINGUISHED, ("distinguish",)),
    (Treatment.FOLLOWED, ("reaffirm", "adhere to", "follow")),
    (Treatment.RELIED_ON, ("relied on", "rely on", "relying on")),
]
_TREATMENT_WINDOW = 180


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    return any(not (span[1] <= c[0] or span[0] >= c[1]) for c in claimed)


def detect_treatment(text: str, at: int, window: int = _TREATMENT_WINDOW) -> Treatment | None:
    ctx = text[max(0, at - window) : at + window].casefold()
    for treatment, keys in _TREATMENT_KEYWORDS:
        if any(k in ctx for k in keys):
            return treatment
    return None


@register(Stage.CITATION_EXTRACTOR, "us_regex")
class USRegexCitationExtractor:
    def __init__(self, case_names: bool = True) -> None:
        self.case_names = case_names

    def _extract_paragraph(self, pid: str, text: str) -> list[CitationMention]:
        mentions: list[CitationMention] = []
        claimed: list[tuple[int, int]] = []
        for canon, rx in _REPORTER_RES:
            for m in rx.finditer(text):
                if _overlaps(m.span(), claimed):
                    continue
                claimed.append(m.span())
                mentions.append(
                    CitationMention(
                        raw=m.group(0),
                        kind=CitationKind.REPORTER,
                        normalized=f"{m.group(1)} {canon} {m.group(2)}",
                        from_para=pid,
                        char_span=m.span(),
                        treatment=detect_treatment(text, m.start()),
                    )
                )
        if self.case_names:
            for m in _CASE_NAME.finditer(text):
                name = f"{m.group(1)} v. {m.group(2)}"
                mentions.append(
                    CitationMention(
                        raw=name,
                        kind=CitationKind.UNKNOWN,
                        normalized=name,
                        from_para=pid,
                        char_span=m.span(),
                        treatment=detect_treatment(text, m.start()),
                    )
                )
        return mentions

    def extract(self, doc: LegalDocument) -> LegalDocument:
        found: list[CitationMention] = []
        for _, para in doc.iter_paragraphs():
            found.extend(self._extract_paragraph(para.pid, para.text))
        doc.citations_out = list(doc.citations_out) + found
        return doc


@register(Stage.CITATION_EXTRACTOR, "eyecite")
class EyeciteExtractor:
    """Free Law Project's eyecite (opt-in). Lazily imported so registering this
    module stays free; instantiation without the extra errors helpfully."""

    def __init__(self) -> None:
        try:
            import eyecite  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "eyecite is required for this extractor. Install with "
                '`pip install -e ".[citations]"`.'
            ) from e

    def extract(self, doc: LegalDocument) -> LegalDocument:  # pragma: no cover - needs extra
        from eyecite import get_citations
        from eyecite.models import CaseCitation

        found: list[CitationMention] = []
        for _, para in doc.iter_paragraphs():
            for cite in get_citations(para.text):
                if not isinstance(cite, CaseCitation):
                    continue
                corrected = cite.corrected_citation()
                found.append(
                    CitationMention(
                        raw=cite.matched_text(),
                        kind=CitationKind.REPORTER,
                        normalized=corrected,
                        from_para=para.pid,
                        treatment=detect_treatment(para.text, cite.index or 0),
                    )
                )
        doc.citations_out = list(doc.citations_out) + found
        return doc
