"""Structural contracts (Protocols) for every pipeline stage.

A new technique is a class that satisfies one of these Protocols and carries a
``@register(stage, name)`` decoration. The pipeline runner resolves components by
name from the experiment config and never needs editing to accommodate a new one
(ADR 0002).

Protocols are structural: implementations need not inherit them, but doing so
lets mypy check the contract. Ingestion-side Protocols (Fetcher, Parser,
CitationExtractor) are declared here for completeness; their implementations live
in jurisdiction packs and are added on the US/PK tracks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from legalrag.core.models import (
    Chunk,
    GeneratedAnswer,
    LegalDocument,
    RetrievalResult,
    ScoredChunk,
    VerificationReport,
)


class Stage:
    """Canonical stage names (registry keys). Strings, so packs can extend."""

    # ingestion
    FETCHER = "fetcher"
    PARSER = "parser"
    CITATION_EXTRACTOR = "citation_extractor"
    # indexing
    CHUNKER = "chunker"
    CONTEXTUALIZER = "contextualizer"
    EMBEDDER = "embedder"
    VECTOR_INDEX = "vector_index"
    LEXICAL_INDEX = "lexical_index"
    # retrieval
    QUERY_TRANSFORMER = "query_transformer"
    FUSION = "fusion"
    RERANKER = "reranker"
    RETRIEVER = "retriever"
    # generation + trust
    ASSEMBLER = "assembler"
    GENERATOR = "generator"
    VERIFIER = "verifier"


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: LegalDocument) -> list[Chunk]: ...


@runtime_checkable
class Contextualizer(Protocol):
    """Annotates chunks with a situating prefix before embedding (E4)."""

    def contextualize(self, doc: LegalDocument, chunks: list[Chunk]) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorIndex(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self, query_vector: Sequence[float], k: int, filters: dict[str, object] | None = None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class LexicalIndex(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(
        self, query_text: str, k: int, filters: dict[str, object] | None = None
    ) -> list[ScoredChunk]: ...


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

@runtime_checkable
class QueryTransformer(Protocol):
    """Produces the search text(s) and metadata filters for a raw query."""

    def transform(self, query: str) -> QueryPlan: ...


class QueryPlan:
    """Lightweight value object (not pydantic — hot path, tiny)."""

    __slots__ = ("filters", "intent", "queries")

    def __init__(
        self,
        queries: list[str],
        filters: dict[str, object] | None = None,
        intent: str | None = None,
    ) -> None:
        self.queries = queries
        self.filters = filters or {}
        self.intent = intent


@runtime_checkable
class Fusion(Protocol):
    def fuse(self, ranked_lists: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Whole retrieval strategy — swappable as one unit while its sub-components
    (embedder, indexes, fusion, reranker) remain individually swappable."""

    def retrieve(self, query: str, k: int) -> RetrievalResult: ...


# --------------------------------------------------------------------------
# Generation + trust
# --------------------------------------------------------------------------

@runtime_checkable
class ContextAssembler(Protocol):
    def assemble(self, query: str, chunks: Sequence[ScoredChunk]) -> AssembledContext: ...


class AssembledContext:
    __slots__ = ("chunks", "text")

    def __init__(self, text: str, chunks: list[ScoredChunk]) -> None:
        self.text = text
        self.chunks = chunks


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, context: AssembledContext) -> GeneratedAnswer: ...


@runtime_checkable
class Verifier(Protocol):
    """Deterministic-first trust stack. Returns the (possibly filtered) answer
    and a report; a citation not in the corpus is never emitted (master plan §5.7)."""

    def verify(self, answer: GeneratedAnswer) -> tuple[GeneratedAnswer, VerificationReport]: ...


# --------------------------------------------------------------------------
# Ingestion (implemented in jurisdiction packs)
# --------------------------------------------------------------------------

@runtime_checkable
class Fetcher(Protocol):
    def fetch(self) -> Iterable[bytes]: ...


@runtime_checkable
class Parser(Protocol):
    def parse(self, raw: bytes, hint: dict[str, object] | None = None) -> LegalDocument: ...


@runtime_checkable
class CitationExtractor(Protocol):
    def extract(self, doc: LegalDocument) -> LegalDocument: ...
