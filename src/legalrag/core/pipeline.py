"""The pipeline runner: builds components from an ExperimentConfig and executes
index -> retrieve -> assemble -> generate -> verify.

This module contains *no* technique-specific logic. It resolves every stage from
the registry by name, so a new technique never edits this file (ADR 0002). The
one composite it wires by hand is hybrid retrieval, because that stage needs live
component objects (embedder + indexes + fusion + reranker) rather than config
params — the sub-components remain individually swappable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from legalrag.core import builtins as _builtins  # noqa: F401  (registers components)
from legalrag.core.config import ExperimentConfig
from legalrag.core.corpus import Corpus, as_json
from legalrag.core.interfaces import (
    AssembledContext,
    ContextAssembler,
    Embedder,
    Fusion,
    Generator,
    LexicalIndex,
    QueryTransformer,
    Reranker,
    Stage,
    VectorIndex,
    Verifier,
)
from legalrag.core.models import GeneratedAnswer, RetrievalResult, ScoredChunk, VerificationReport
from legalrag.core.registry import build


@dataclass(slots=True)
class AnswerBundle:
    retrieval: RetrievalResult
    answer: GeneratedAnswer
    report: VerificationReport


class Pipeline:
    def __init__(self, config: ExperimentConfig, corpus: Corpus) -> None:
        self.config = config
        self.corpus = corpus

        ix, rt, gen = config.index, config.retrieval, config.generation
        self.chunker = build(Stage.CHUNKER, ix.chunker)
        self.contextualizer = (
            None
            if ix.contextualizer.name == "none"
            else build(Stage.CONTEXTUALIZER, ix.contextualizer)
        )
        self.embedder: Embedder = build(Stage.EMBEDDER, ix.embedder)
        self.vector_index: VectorIndex = build(Stage.VECTOR_INDEX, ix.vector_index)
        self.lexical_index: LexicalIndex = build(Stage.LEXICAL_INDEX, ix.lexical_index)

        self.query_transformer: QueryTransformer = build(
            Stage.QUERY_TRANSFORMER, rt.query_transformer
        )
        self.fusion: Fusion = build(Stage.FUSION, rt.fusion)
        self.reranker: Reranker = build(Stage.RERANKER, rt.reranker)

        self.assembler: ContextAssembler = build(Stage.ASSEMBLER, gen.assembler)
        self.generator: Generator = build(Stage.GENERATOR, gen.generator)
        self.verifier: Verifier = build(Stage.VERIFIER, gen.verifier, corpus=corpus)

        self.index_id: str | None = None

    # --- indexing -------------------------------------------------------

    def build_index(self) -> str:
        h = hashlib.sha256()
        n_chunks = 0
        for doc in self.corpus:
            chunks = self.chunker.chunk(doc)
            if self.contextualizer is not None:
                chunks = self.contextualizer.contextualize(doc, chunks)
            if not chunks:
                continue
            vectors = self.embedder.embed_documents([c.embed_text() for c in chunks])
            self.vector_index.add(chunks, vectors)
            self.lexical_index.add(chunks)
            n_chunks += len(chunks)
            for c in chunks:
                h.update(c.chunk_id.encode("utf-8"))
        self.index_id = (
            f"{self.embedder.model_id}:{n_chunks}:{h.hexdigest()[:12]}"
        )
        return self.index_id

    # --- retrieval ------------------------------------------------------

    def retrieve(self, query: str, qid: str = "") -> RetrievalResult:
        rt = self.config.retrieval
        plan = self.query_transformer.transform(query)
        ranked_lists: list[list[ScoredChunk]] = []
        for sub in plan.queries:
            if rt.dense_k > 0:
                qv = self.embedder.embed_query(sub)
                ranked_lists.append(
                    self.vector_index.search(qv, rt.dense_k, plan.filters)
                )
            if rt.sparse_k > 0:
                ranked_lists.append(
                    self.lexical_index.search(sub, rt.sparse_k, plan.filters)
                )
        fused = self.fusion.fuse(ranked_lists) if ranked_lists else []
        reranked = self.reranker.rerank(query, fused, rt.rerank_top_k)
        return RetrievalResult(qid=qid, scored=reranked[: rt.final_k])

    # --- full answer ----------------------------------------------------

    def answer(self, query: str, qid: str = "") -> AnswerBundle:
        retrieval = self.retrieve(query, qid=qid)
        context: AssembledContext = self.assembler.assemble(query, retrieval.scored)
        raw = self.generator.generate(query, context)
        raw = raw.model_copy(update={"qid": qid})
        verified, report = self.verifier.verify(raw)
        return AnswerBundle(retrieval=retrieval, answer=verified, report=report)

    # --- bookkeeping ----------------------------------------------------

    def signature(self) -> str:
        return as_json(
            {
                "config_hash": self.config.config_hash(),
                "index_id": self.index_id,
                "embedder": self.embedder.model_id,
            }
        )
