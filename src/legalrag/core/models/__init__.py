"""Canonical data model (master plan §4). Import types from here."""

from __future__ import annotations

from legalrag.core.models.chunk import Chunk
from legalrag.core.models.document import (
    Citation,
    CitationMention,
    CourtRef,
    DocVersion,
    LegalDocument,
    Paragraph,
    Provenance,
    QualityReport,
    Section,
)
from legalrag.core.models.enums import (
    NEGATIVE_TREATMENTS,
    CitationKind,
    CourtLevel,
    DocType,
    LicenseBasis,
    QueryType,
    Treatment,
)
from legalrag.core.models.results import (
    AnswerCitation,
    CheckResult,
    EvalQuery,
    GeneratedAnswer,
    Qrel,
    RetrievalResult,
    RunManifest,
    ScoredChunk,
    VerificationReport,
)

__all__ = [
    "NEGATIVE_TREATMENTS",
    "AnswerCitation",
    "CheckResult",
    "Chunk",
    "Citation",
    "CitationKind",
    "CitationMention",
    "CourtLevel",
    "CourtRef",
    "DocType",
    "DocVersion",
    "EvalQuery",
    "GeneratedAnswer",
    "LegalDocument",
    "LicenseBasis",
    "Paragraph",
    "Provenance",
    "Qrel",
    "QualityReport",
    "QueryType",
    "RetrievalResult",
    "RunManifest",
    "ScoredChunk",
    "Section",
    "Treatment",
    "VerificationReport",
]
