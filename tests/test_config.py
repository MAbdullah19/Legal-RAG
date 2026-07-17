"""Experiment config loading + hashing (ADR 0002)."""

from __future__ import annotations

from legalrag.core.config import ExperimentConfig, load_experiment
from legalrag.core.registry import ComponentSpec


def test_load_e0_and_e1() -> None:
    e0 = load_experiment("e0_naive_baseline")
    e1 = load_experiment("e1_hybrid")
    assert e0.name == "e0_naive_baseline"
    assert e0.corpus == "fixture"
    # E0 is dense-only; E1 turns on lexical
    assert e0.retrieval.sparse_k == 0
    assert e1.retrieval.sparse_k > 0


def test_specs_are_coerced() -> None:
    e0 = load_experiment("e0_naive_baseline")
    assert isinstance(e0.index.chunker, ComponentSpec)
    assert e0.index.chunker.name == "fixed"
    assert e0.index.chunker.params["max_tokens"] == 220


def test_config_hash_is_stable_and_sensitive() -> None:
    e0a = load_experiment("e0_naive_baseline")
    e0b = load_experiment("e0_naive_baseline")
    e1 = load_experiment("e1_hybrid")
    assert e0a.config_hash() == e0b.config_hash()
    assert e0a.config_hash() != e1.config_hash()


def test_defaults_fill_in() -> None:
    cfg = ExperimentConfig(name="x", corpus="fixture")
    assert cfg.index.embedder.name == "hashing"
    assert cfg.generation.verifier.name == "deterministic"
