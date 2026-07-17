"""Generators (master plan §5.6).

``extractive`` is deterministic and LLM-free: it quotes verbatim from the top
retrieved paragraphs and attaches a pinpoint citation per quote. This lets the
whole grounded-generation + verification path run in CI with no API key, and it
is by construction citation-faithful (every quote is a substring of a real source
paragraph). The ``anthropic`` / ``local`` grounded generators register here and
are selected in config for quality experiments; they emit the same
GeneratedAnswer shape, so the verifier and metrics are identical.
"""

from __future__ import annotations

from legalrag.core.interfaces import AssembledContext, Stage
from legalrag.core.models import AnswerCitation, GeneratedAnswer
from legalrag.core.registry import register
from legalrag.core.text import first_sentence


@register(Stage.GENERATOR, "extractive")
class ExtractiveGenerator:
    def __init__(self, max_sources: int = 3) -> None:
        self.max_sources = max_sources

    def generate(self, query: str, context: AssembledContext) -> GeneratedAnswer:
        chunks = context.chunks[: self.max_sources]
        if not chunks:
            return GeneratedAnswer(
                qid="",
                text="I could not find supporting authority in the corpus for this question.",
                abstained=True,
            )
        citations: list[AnswerCitation] = []
        sentences: list[str] = []
        for i, sc in enumerate(chunks, start=1):
            quote = first_sentence(sc.text)
            if not quote:
                continue
            citations.append(
                AnswerCitation(doc_id=sc.doc_id, para_ids=list(sc.para_ids), quote=quote)
            )
            sentences.append(f'[S{i}] "{quote}"')
        body = " ".join(sentences)
        text = f"Based on the retrieved authorities: {body}"
        return GeneratedAnswer(
            qid="",
            text=text,
            citations=citations,
            used_chunk_ids=[sc.chunk_id for sc in chunks],
        )
