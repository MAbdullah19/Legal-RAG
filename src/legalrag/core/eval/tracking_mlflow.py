"""MLflow-backed run tracker (master plan §6.4).

The engine-side adapter for the MLflow service defined in
``infra/docker-compose.yml``. It records the SAME ``RunManifest`` + ``EvalReport``
the JSON tracker does, so switching trackers is an env change, not a change to the
eval loop — every experiment lands in one comparable history (docs/deployment.md,
the two-PC pattern).

MLflow is a **lazy import** (extra: ``.[tracking]``), so this module never touches
the E0 smoke path or CI (ADR 0003) — importing it costs nothing and needs no
server. The two translations that MLflow constrains (params must be short
strings; metrics must be finite floats) live here as pure functions and are
unit-tested without a server. Integration is validated against a tracking URI:
``file:./mlruns`` locally (no service), or the docker MLflow over the LAN.
"""

from __future__ import annotations

import math
import os

from legalrag.core.eval.harness import EvalReport
from legalrag.core.eval.tracking import _REPO_ROOT, JSONRunTracker
from legalrag.core.models import RunManifest

# MLflow rejects params longer than this and non-finite metric values; the
# mapping below is defensive so a run never fails on a logging technicality.
_MAX_PARAM_LEN = 500


def params_from_manifest(manifest: RunManifest) -> dict[str, str]:
    """Flatten a manifest into MLflow params (all string-valued, length-capped)."""
    params: dict[str, str] = {
        "experiment": manifest.experiment,
        "config_hash": manifest.config_hash,
        "corpus_id": manifest.corpus_id,
        "seed": str(manifest.seed),
    }
    if manifest.index_id:
        params["index_id"] = manifest.index_id
    if manifest.git_commit:
        params["git_commit"] = manifest.git_commit
    for name, version in manifest.model_versions.items():
        params[f"model.{name}"] = version
    return {k: str(v)[:_MAX_PARAM_LEN] for k, v in params.items()}


def metrics_from_report(report: EvalReport) -> dict[str, float]:
    """Keep only finite numeric metrics — MLflow rejects NaN/inf, which the
    harness emits for metric groups a corpus doesn't exercise (e.g. no
    unanswerable queries)."""
    out: dict[str, float] = {}
    for name, value in report.metrics.items():
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """$MLFLOW_TRACKING_URI when set, else a repo-local ``file:`` store that needs
    no server — so this tracker works offline and only points at the docker
    service (or the 3080 box over LAN) when the env var is present."""
    return (
        tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"file:{(_REPO_ROOT / 'mlruns').as_posix()}"
    )


class MLflowRunTracker:
    """Logs runs to an MLflow tracking server or a local file store.

    Composes the JSON tracker: the local ``runs/`` manifest is still written
    (so reproducibility artifacts never depend on a service being up) and is then
    attached to the MLflow run, while params/metrics are logged for cross-run
    comparison. ``log`` returns the local run dir, matching ``JSONRunTracker`` so
    callers are interchangeable.
    """

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment: str = "legalrag",
        also_json: bool = True,
    ) -> None:
        self.tracking_uri = resolve_tracking_uri(tracking_uri)
        self.experiment = experiment
        self._json = JSONRunTracker() if also_json else None

    def log(self, manifest: RunManifest, report: EvalReport) -> str:
        import mlflow

        run_dir = self._json.log(manifest, report) if self._json else ""

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        run_name = f"{manifest.experiment}:{manifest.config_hash}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params_from_manifest(manifest))
            mlflow.log_metrics(metrics_from_report(report))
            mlflow.set_tag("git_commit", manifest.git_commit or "unknown")
            mlflow.set_tag("corpus_id", manifest.corpus_id)
            if run_dir:
                # attach the exact JSON manifest / report / per-query records
                mlflow.log_artifacts(run_dir, artifact_path="run")
        return run_dir or run_name
