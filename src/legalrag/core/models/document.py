"""The canonical, jurisdiction-neutral document model (master plan §4).

This is the contract that keeps every downstream component jurisdiction-agnostic
and makes the multi-jurisdiction track (RQ5) possible. Jurisdiction packs supply
parsers/extractors that *populate* these types; nothing here knows about a
specific legal system.

Design commitments:
- The **paragraph is the atomic, pinpoint-citable unit**. Answers cite
  ``doc_id ¶ pid`` so every claim is human-verifiable.
- **Citations are first-class**, not free text — they become the substrate for
  the citation graph and the deterministic citation verifier.
- **Statutes are version chains** (:class:`DocVersion`); queries carry an as-of date.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legalrag.core.models.enums import (
    CitationKind,
    CourtLevel,
    DocType,
    LicenseBasis,
    Treatment,
)


class _Model(BaseModel):
    """Strict base: unknown fields are errors, so parser bugs surface loudly."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class CourtRef(_Model):
    court_id: str            # canonical id from a pack's court registry, e.g. "pk/sc"
    name: str
    level: CourtLevel
    jurisdiction: str        # "us", "pk", ...


class Citation(_Model):
    """A citation *string* — how a document is (or can be) referred to."""

    raw: str
    kind: CitationKind = CitationKind.UNKNOWN
    normalized: str | None = None
    reporter: str | None = None       # "PLD", "SCMR", "U.S."
    year: int | None = None
    volume: str | None = None
    page: str | None = None

    def key(self) -> str:
        """Stable comparison key used by the citation verifier and graph."""
        return (self.normalized or self.raw).strip().casefold()


class CitationMention(_Model):
    """An outbound citation occurring inside a document's body."""

    raw: str
    kind: CitationKind = CitationKind.UNKNOWN
    normalized: str | None = None
    from_para: str | None = None       # pid of the paragraph containing the mention
    char_span: tuple[int, int] | None = None
    target_doc_id: str | None = None   # resolved target, or None if unresolved
    treatment: Treatment | None = None

    def key(self) -> str:
        return (self.normalized or self.raw).strip().casefold()


class Provenance(_Model):
    source: str                         # connector name, e.g. "us/cap"
    url: str | None = None
    fetched_at: datetime | None = None
    checksum: str | None = None         # sha256 of raw bytes
    license_basis: LicenseBasis = LicenseBasis.UNKNOWN
    license_note: str | None = None


class DocVersion(_Model):
    """Point-in-time chain for statutes/rules."""

    version_label: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    predecessor_doc_id: str | None = None


class QualityReport(_Model):
    is_ocr: bool = False
    ocr_confidence: float | None = None   # 0..1, mean page confidence
    parse_score: float | None = None      # 0..1, structural sanity
    dedup_cluster_id: str | None = None   # docs sharing this id are the same judgment
    warnings: list[str] = Field(default_factory=list)


class Paragraph(_Model):
    """Atomic pinpoint unit. ``pid`` is stable within a document."""

    pid: str = ""                         # assigned "p0001".. if left blank
    para_no: str | None = None            # human-facing number, e.g. "12"
    text: str
    page_span: tuple[int, int] | None = None

    def is_empty(self) -> bool:
        return not self.text.strip()


class Section(_Model):
    heading: str = ""
    path: list[str] = Field(default_factory=list)   # breadcrumb, e.g. ["Facts"]
    paragraphs: list[Paragraph] = Field(default_factory=list)


class LegalDocument(_Model):
    """A judgment, order, statute, or rule in canonical form."""

    doc_id: str                           # "{jurisdiction}/{court}/{id}/{version}"
    doc_type: DocType
    jurisdiction: str
    title: str
    court: CourtRef | None = None
    case_numbers: list[str] = Field(default_factory=list)
    citations_self: list[Citation] = Field(default_factory=list)
    judges: list[str] = Field(default_factory=list)
    decision_date: date | None = None
    language: str = "en"
    version: DocVersion | None = None
    source: Provenance
    body: list[Section] = Field(default_factory=list)
    citations_out: list[CitationMention] = Field(default_factory=list)
    quality: QualityReport = Field(default_factory=QualityReport)

    @model_validator(mode="after")
    def _assign_and_check_pids(self) -> LegalDocument:
        seen: set[str] = set()
        counter = 0
        for section in self.body:
            for para in section.paragraphs:
                counter += 1
                if not para.pid:
                    para.pid = f"p{counter:04d}"
                if para.pid in seen:
                    raise ValueError(f"duplicate paragraph id {para.pid!r} in {self.doc_id}")
                seen.add(para.pid)
        return self

    # --- convenience accessors used across the pipeline ---

    def iter_paragraphs(self) -> Iterator[tuple[Section, Paragraph]]:
        for section in self.body:
            for para in section.paragraphs:
                yield section, para

    def paragraphs(self) -> list[Paragraph]:
        return [p for _, p in self.iter_paragraphs()]

    def get_paragraph(self, pid: str) -> Paragraph | None:
        for _, para in self.iter_paragraphs():
            if para.pid == pid:
                return para
        return None

    def full_text(self) -> str:
        return "\n".join(p.text for _, p in self.iter_paragraphs())

    def pinpoint(self, pid: str) -> str:
        """Human-facing pinpoint, e.g. ``A v. B ¶ 12``."""
        para = self.get_paragraph(pid)
        num = para.para_no if para and para.para_no else pid
        return f"{self.title} ¶ {num}"

    def authority_weight(self) -> float:
        return self.court.level.authority_weight if self.court else CourtLevel.OTHER.authority_weight
