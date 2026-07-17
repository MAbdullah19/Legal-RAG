"""Chunkers (ablation axis A, master plan §5.2).

E0 uses ``fixed``; ``structure_aware`` (never splits across sections, never
splits a numbered paragraph) is the E3 contender. Both attach filter payload
(court level, date, doc_type, jurisdiction) so retrieval can enforce metadata
filters — a legal-IR necessity.
"""

from __future__ import annotations

from collections.abc import Iterable

from legalrag.core.interfaces import Stage
from legalrag.core.models import Chunk, LegalDocument, Paragraph, Section
from legalrag.core.registry import register
from legalrag.core.text import estimate_tokens


def _payload(doc: LegalDocument, section: Section) -> dict[str, object]:
    return {
        "jurisdiction": doc.jurisdiction,
        "doc_type": str(doc.doc_type),
        "court_level": str(doc.court.level) if doc.court else None,
        "decision_date": doc.decision_date.isoformat() if doc.decision_date else None,
        "section": section.heading or None,
    }


def _make_chunk(
    doc: LegalDocument, section: Section, group: list[Paragraph]
) -> Chunk:
    pids = [p.pid for p in group]
    text = "\n".join(p.text for p in group)
    return Chunk(
        chunk_id=f"{doc.doc_id}#{pids[0]}-{pids[-1]}",
        doc_id=doc.doc_id,
        para_ids=pids,
        text=text,
        token_estimate=estimate_tokens(text),
        payload=_payload(doc, section),
    )


def _pack(
    doc: LegalDocument, section: Section, paras: Iterable[Paragraph], max_tokens: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    group: list[Paragraph] = []
    running = 0
    for para in paras:
        if para.is_empty():
            continue
        t = estimate_tokens(para.text)
        if group and running + t > max_tokens:
            chunks.append(_make_chunk(doc, section, group))
            group, running = [], 0
        group.append(para)
        running += t
    if group:
        chunks.append(_make_chunk(doc, section, group))
    return chunks


@register(Stage.CHUNKER, "fixed")
class FixedTokenChunker:
    """Greedy fixed-size packing over the flat paragraph stream (E0 baseline).

    Paragraphs are never split mid-way (pinpoint integrity); a single oversized
    paragraph simply becomes its own chunk.
    """

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def chunk(self, doc: LegalDocument) -> list[Chunk]:
        # flatten across sections, tagging each paragraph's owning section
        chunks: list[Chunk] = []
        group: list[Paragraph] = []
        group_section: Section | None = None
        running = 0
        for section, para in doc.iter_paragraphs():
            if para.is_empty():
                continue
            t = estimate_tokens(para.text)
            if group and running + t > self.max_tokens:
                assert group_section is not None
                chunks.append(_make_chunk(doc, group_section, group))
                group, running = [], 0
            if not group:
                group_section = section
            group.append(para)
            running += t
        if group and group_section is not None:
            chunks.append(_make_chunk(doc, group_section, group))
        return chunks


@register(Stage.CHUNKER, "structure_aware")
class StructureAwareChunker:
    """Chunks within a section only — never merges paragraphs across sections,
    so a chunk always carries a coherent structural context (E3 contender)."""

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def chunk(self, doc: LegalDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in doc.body:
            chunks.extend(_pack(doc, section, section.paragraphs, self.max_tokens))
        return chunks
