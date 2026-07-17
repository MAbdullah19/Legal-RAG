"""Graph-expanded retrieval (E7) and the external embedder adapters."""

from __future__ import annotations

import pytest

from legalrag.core.builtins import _embedders_external, _graph_expand  # noqa: F401
from legalrag.core.config import load_experiment
from legalrag.core.corpus import load_corpus, load_eval_queries
from legalrag.core.eval.harness import evaluate
from legalrag.core.interfaces import Stage
from legalrag.core.pipeline import Pipeline
from legalrag.core.registry import REGISTRY, RegistryError


def _pipeline(experiment: str) -> Pipeline:
    cfg = load_experiment(experiment)
    p = Pipeline(cfg, load_corpus(cfg.corpus))
    p.build_index()
    return p


def test_graph_expander_registered() -> None:
    assert "none" in REGISTRY.available(Stage.GRAPH_EXPANDER)
    assert "citation_neighbors" in REGISTRY.available(Stage.GRAPH_EXPANDER)


def test_e7_injects_precedent_neighbour_that_text_retrieval_missed() -> None:
    # q6: the foundational case D4 is relevant but is not a text match for the
    # query; E1 misses it, E7 reaches it through the citation graph (D6/D5 -> D4).
    q6 = "is Zephyr Textiles still good law on remoteness overruled"
    e1_rank = _pipeline("e1_hybrid").retrieve(q6).doc_ranking()
    e7_rank = _pipeline("e7_graph").retrieve(q6).doc_ranking()
    foundational = "xx/apex/1990-003/v1"
    assert foundational not in e1_rank        # text retrieval alone misses it
    assert foundational in e7_rank            # graph expansion recovers it


def test_e7_improves_recall_without_regressing_or_hallucinating() -> None:
    def report(name: str):
        p = _pipeline(name)
        cfg = p.config
        return evaluate(p, load_eval_queries(cfg.corpus), ks=cfg.eval.ks, ndcg_k=cfg.eval.ndcg_k)

    e1, e7 = report("e1_hybrid"), report("e7_graph")
    # graph expansion lifts precedent-chain recall to complete, at the same budget
    assert e7.by_type["multi_hop"]["recall@10"] == 1.0
    assert e7.by_type["multi_hop"]["recall@10"] > e1.by_type["multi_hop"]["recall@10"]
    # and does not regress aggregate recall or introduce fabricated citations
    assert e7.metrics["recall@10"] >= e1.metrics["recall@10"]
    assert e7.metrics["hallucinated_citation_rate"] == 0.0


def test_e7_injected_citations_all_resolve() -> None:
    p = _pipeline("e7_graph")
    for q in load_eval_queries(p.config.corpus):
        bundle = p.answer(q.text, qid=q.qid)
        for c in bundle.answer.citations:
            assert c.doc_id in p.corpus  # injected neighbours are real docs


def test_e0_config_unaffected_by_new_stage() -> None:
    # backward-compat: E0 doesn't mention graph_expander -> defaults to none
    cfg = load_experiment("e0_naive_baseline")
    assert cfg.retrieval.graph_expander.name == "none"


def test_external_embedders_error_helpfully_without_extra() -> None:
    # sentence-transformers / voyageai are not installed in the core/dev env;
    # instantiation must fail with an actionable message, not an obscure one.
    with pytest.raises((ImportError, RuntimeError)) as exc:
        REGISTRY.build(Stage.EMBEDDER, {"name": "sentence_transformer"})
    assert "pip install" in str(exc.value)


def test_unknown_still_raises_registry_error() -> None:
    with pytest.raises(RegistryError):
        REGISTRY.build(Stage.EMBEDDER, {"name": "does_not_exist"})
