"""End-to-end pipeline test on the fixture corpus (the P0 exit criterion:
E0 runs end-to-end and produces a tracked, sane result)."""

from __future__ import annotations

import pytest

from legalrag.core.builtins import _chunkers  # noqa: F401  ensure registration
from legalrag.core.config import load_experiment
from legalrag.core.corpus import load_corpus, load_eval_queries
from legalrag.core.eval.harness import evaluate
from legalrag.core.interfaces import Stage
from legalrag.core.pipeline import Pipeline
from legalrag.core.registry import REGISTRY


@pytest.fixture(scope="module")
def e0_pipeline() -> Pipeline:
    cfg = load_experiment("e0_naive_baseline")
    corpus = load_corpus(cfg.corpus)
    p = Pipeline(cfg, corpus)
    p.build_index()
    return p


def test_builtins_registered() -> None:
    assert "fixed" in REGISTRY.available(Stage.CHUNKER)
    assert "hashing" in REGISTRY.available(Stage.EMBEDDER)
    assert "rrf" in REGISTRY.available(Stage.FUSION)
    assert "deterministic" in REGISTRY.available(Stage.VERIFIER)


def test_index_built(e0_pipeline: Pipeline) -> None:
    assert e0_pipeline.index_id is not None
    assert len(e0_pipeline.corpus) == 14


def test_retrieval_finds_expected_doc(e0_pipeline: Pipeline) -> None:
    # conceptual bail query should surface the bail judgment at rank 1
    result = e0_pipeline.retrieve("principles for granting bail in economic offences")
    assert result.doc_ranking()[0] == "xx/apex/2018-050/v1"


def test_answer_is_grounded_and_verified(e0_pipeline: Pipeline) -> None:
    bundle = e0_pipeline.answer("damages for remoteness in breach of contract")
    assert bundle.answer.citations, "expected at least one grounded citation"
    # every emitted citation resolves in the corpus (no hallucination)
    for c in bundle.answer.citations:
        assert c.doc_id in e0_pipeline.corpus
        doc = e0_pipeline.corpus.get(c.doc_id)
        assert doc is not None
        for pid in c.para_ids:
            assert doc.get_paragraph(pid) is not None
    assert bundle.report.passed


def test_full_eval_zero_hallucination(e0_pipeline: Pipeline) -> None:
    cfg = e0_pipeline.config
    queries = load_eval_queries(cfg.corpus)
    report = evaluate(e0_pipeline, queries, ks=cfg.eval.ks, ndcg_k=cfg.eval.ndcg_k)
    assert report.n_queries == 10
    assert report.metrics["hallucinated_citation_rate"] == 0.0
    # the fixture is easy enough that a sane baseline clears these floors
    assert report.metrics["recall@10"] >= 0.8
    assert report.metrics["mrr"] >= 0.6
    # per-query-type breakdown is populated (legal IR reports per type)
    assert "citation_lookup" in report.by_type


def test_unanswerable_query_makes_no_false_citation(e0_pipeline: Pipeline) -> None:
    # q9 has no supporting doc; whatever happens, no fabricated citation may appear
    bundle = e0_pipeline.answer("penalty for cryptocurrency fraud under the Digital Assets Act")
    for c in bundle.answer.citations:
        assert c.doc_id in e0_pipeline.corpus
