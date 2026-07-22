"""Run tracking (master plan §3.2, §6.4).

Every run records a manifest — config hash, corpus id, index id, git commit,
seed, model versions, metrics — so any result is reproducible. The default sink
is JSON on disk (dependency-free, CI-friendly); an MLflow-backed tracker
registers the same interface later.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from legalrag.core.eval.harness import EvalReport, report_to_dict
from legalrag.core.models import RunManifest

_REPO_ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = _REPO_ROOT / "runs"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


class RunTracker(Protocol):
    def log(self, manifest: RunManifest, report: EvalReport) -> str: ...


class JSONRunTracker:
    def __init__(self, root: Path = RUNS_DIR) -> None:
        self.root = root

    def log(self, manifest: RunManifest, report: EvalReport) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / f"{ts}_{manifest.experiment}_{manifest.config_hash}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "report.json").write_text(
            json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "per_query.jsonl").write_text(
            "\n".join(json.dumps(asdict(r)) for r in report.records),
            encoding="utf-8",
        )
        return str(run_dir)


def resolve_tracker() -> RunTracker:
    """Pick a tracker from the environment, never failing the run.

    MLflow when ``MLFLOW_TRACKING_URI`` is set *and* the ``[tracking]`` extra is
    installed; otherwise the dependency-free JSON tracker. If MLflow is requested
    but unavailable (extra not installed), we fall back to JSON rather than crash —
    the local ``runs/`` manifest is always written either way.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        import importlib.util

        if importlib.util.find_spec("mlflow") is not None:
            from legalrag.core.eval.tracking_mlflow import MLflowRunTracker

            return MLflowRunTracker()
    return JSONRunTracker()


def build_manifest(
    *,
    experiment: str,
    config_hash: str,
    corpus_id: str,
    index_id: str | None,
    seed: int,
    model_versions: dict[str, str],
    metrics: dict[str, float],
) -> RunManifest:
    return RunManifest(
        experiment=experiment,
        config_hash=config_hash,
        corpus_id=corpus_id,
        index_id=index_id,
        git_commit=git_commit(),
        seed=seed,
        model_versions=model_versions,
        metrics=metrics,
    )
