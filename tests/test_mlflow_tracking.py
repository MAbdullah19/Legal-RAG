"""MLflow tracker — pure mapping + factory selection, hermetic (no mlflow, no server).

The mlflow import lives inside ``MLflowRunTracker.log``, so the manifest->params
and report->metrics translation is unit-testable on its own. Integration against a
tracking URI (file:./mlruns or the docker MLflow) is validated when the extra is
installed (docs/deployment.md)."""

from __future__ import annotations

import math

from legalrag.core.eval.harness import EvalReport
from legalrag.core.eval.tracking import JSONRunTracker, resolve_tracker
from legalrag.core.eval.tracking_mlflow import (
    metrics_from_report,
    params_from_manifest,
    resolve_tracking_uri,
)
from legalrag.core.models import RunManifest


def _manifest() -> RunManifest:
    return RunManifest(
        experiment="e9_local_grounded",
        config_hash="abc123",
        corpus_id="fixture",
        index_id="idx-1",
        git_commit="deadbee",
        seed=7,
        model_versions={"embedder": "bge-small-en-v1.5", "reranker": "ms-marco-MiniLM"},
        metrics={"recall@10": 0.9},
    )


def _report() -> EvalReport:
    return EvalReport(
        experiment="e9_local_grounded",
        n_queries=3,
        metrics={
            "recall@10": 0.889,
            "ndcg@10": 0.785,
            "hallucinated_citation_rate": 0.0,
            "over_abstention_rate": float("nan"),  # dropped: no answerable queries
        },
        by_type={},
    )


def test_params_are_string_valued_and_include_models() -> None:
    p = params_from_manifest(_manifest())
    assert p["experiment"] == "e9_local_grounded"
    assert p["seed"] == "7"  # coerced to str
    assert p["git_commit"] == "deadbee"
    assert p["model.embedder"] == "bge-small-en-v1.5"
    assert p["model.reranker"] == "ms-marco-MiniLM"
    assert all(isinstance(v, str) for v in p.values())


def test_metrics_drop_non_finite() -> None:
    m = metrics_from_report(_report())
    assert m["recall@10"] == 0.889
    assert m["hallucinated_citation_rate"] == 0.0
    assert "over_abstention_rate" not in m  # NaN filtered
    assert all(math.isfinite(v) for v in m.values())


def test_tracking_uri_defaults_to_local_file_store(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri().startswith("file:")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://192.168.1.20:5000")
    assert resolve_tracking_uri() == "http://192.168.1.20:5000"
    assert resolve_tracking_uri("http://explicit:5000") == "http://explicit:5000"


def test_factory_uses_json_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert isinstance(resolve_tracker(), JSONRunTracker)
