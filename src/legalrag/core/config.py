"""Experiment configuration schema + loader (ADR 0002).

A full RAG system instance is a validated value here, not code. Ablations are
config diffs; every run records this config's hash so results are reproducible.
Kept framework-independent (plain pydantic + YAML); Hydra may later wrap this for
multirun sweeps without changing the schema.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from legalrag.core.registry import ComponentSpec

Spec = Annotated[ComponentSpec, BeforeValidator(ComponentSpec.from_obj)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexConfig(_Model):
    chunker: Spec = Field(default_factory=lambda: ComponentSpec(name="fixed"))
    contextualizer: Spec = Field(default_factory=lambda: ComponentSpec(name="none"))
    embedder: Spec = Field(default_factory=lambda: ComponentSpec(name="hashing"))
    vector_index: Spec = Field(default_factory=lambda: ComponentSpec(name="memory"))
    lexical_index: Spec = Field(default_factory=lambda: ComponentSpec(name="bm25"))


class RetrievalConfig(_Model):
    query_transformer: Spec = Field(default_factory=lambda: ComponentSpec(name="none"))
    dense_k: int = 20
    sparse_k: int = 0                # 0 disables lexical (E0 is dense-only)
    fusion: Spec = Field(default_factory=lambda: ComponentSpec(name="rrf"))
    reranker: Spec = Field(default_factory=lambda: ComponentSpec(name="none"))
    rerank_top_k: int = 10
    final_k: int = 10


class GenerationConfig(_Model):
    assembler: Spec = Field(default_factory=lambda: ComponentSpec(name="concat"))
    generator: Spec = Field(default_factory=lambda: ComponentSpec(name="extractive"))
    verifier: Spec = Field(default_factory=lambda: ComponentSpec(name="deterministic"))


class EvalConfig(_Model):
    ks: list[int] = Field(default_factory=lambda: [5, 10, 20])
    ndcg_k: int = 10
    bootstrap_samples: int = 1000


class ExperimentConfig(_Model):
    name: str
    corpus: str
    seed: int = 0
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        blob = repr(sorted(_flatten(payload).items())).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = _REPO_ROOT / "experiments"


def load_experiment(name_or_path: str) -> ExperimentConfig:
    """Load by experiment name (``e0_naive_baseline`` -> experiments/e0_naive_baseline.yaml)
    or by explicit path."""
    path = Path(name_or_path)
    if not path.exists():
        path = EXPERIMENTS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"experiment config not found: {name_or_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(data)
