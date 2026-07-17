"""``legalrag`` command line (master plan §3.3).

Commands map to the research loop: inspect registered components, build an index,
probe a query ad hoc, and run a tracked evaluation. ``ingest`` is a placeholder
that jurisdiction packs fill in on the US/PK tracks.
"""

from __future__ import annotations

import math

import typer
from rich.console import Console
from rich.table import Table

from legalrag.core import builtins as _builtins  # noqa: F401  (register components)
from legalrag.core.config import load_experiment
from legalrag.core.corpus import load_corpus, load_eval_queries
from legalrag.core.eval.harness import evaluate
from legalrag.core.eval.tracking import JSONRunTracker, build_manifest
from legalrag.core.models import EvalQuery
from legalrag.core.pipeline import Pipeline
from legalrag.core.registry import REGISTRY

app = typer.Typer(add_completion=False, help="Advanced RAG over judicial corpora.")
console = Console()


def _fmt(v: float) -> str:
    return "  -  " if isinstance(v, float) and math.isnan(v) else f"{v:.4f}"


def _build_pipeline(experiment: str) -> tuple[Pipeline, list[EvalQuery]]:
    cfg = load_experiment(experiment)
    corpus = load_corpus(cfg.corpus)
    pipeline = Pipeline(cfg, corpus)
    pipeline.build_index()
    queries = load_eval_queries(cfg.corpus)
    return pipeline, queries


@app.command()
def components() -> None:
    """List every registered component by stage."""
    table = Table(title="Registered components")
    table.add_column("stage", style="cyan")
    table.add_column("names", style="green")
    for stage in REGISTRY.stages():
        table.add_row(stage, ", ".join(REGISTRY.available(stage)))
    console.print(table)


@app.command()
def query(
    text: str,
    experiment: str = typer.Option("e0_naive_baseline", "--experiment", "-e"),
) -> None:
    """Answer a single query ad hoc (no tracking)."""
    pipeline, _ = _build_pipeline(experiment)
    bundle = pipeline.answer(text)
    console.rule("Answer")
    console.print(bundle.answer.text)
    if bundle.answer.citations:
        console.rule("Citations (verified)")
        for c in bundle.answer.citations:
            doc = pipeline.corpus.get(c.doc_id)
            pin = doc.pinpoint(c.para_ids[0]) if doc and c.para_ids else c.doc_id
            console.print(f"• {pin}  [{c.doc_id}]")
    if not bundle.report.passed:
        console.rule("[red]Verifier flags")
        for chk in bundle.report.failures():
            console.print(f"[red]{chk.name}[/]: {', '.join(chk.offending)}")


@app.command()
def index(experiment: str = typer.Option("e0_naive_baseline", "--experiment", "-e")) -> None:
    """Build the index for an experiment and report its id."""
    pipeline, _ = _build_pipeline(experiment)
    console.print(f"index_id = [green]{pipeline.index_id}[/]  ({len(pipeline.corpus)} docs)")


@app.command("eval")
def eval_cmd(
    experiment: str = typer.Option("e0_naive_baseline", "--experiment", "-e"),
    smoke: bool = typer.Option(False, "--smoke", help="assert P0 exit criteria and exit nonzero on failure"),
) -> None:
    """Run a tracked evaluation over the experiment's corpus eval set."""
    cfg = load_experiment(experiment)
    corpus = load_corpus(cfg.corpus)
    pipeline = Pipeline(cfg, corpus)
    pipeline.build_index()
    queries = load_eval_queries(cfg.corpus)
    if not queries:
        console.print(f"[yellow]no eval queries for corpus {cfg.corpus!r}[/]")
        raise typer.Exit(code=1 if smoke else 0)

    report = evaluate(pipeline, queries, ks=cfg.eval.ks, ndcg_k=cfg.eval.ndcg_k)

    table = Table(title=f"{cfg.name}  ·  corpus={cfg.corpus}  ·  n={report.n_queries}")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right", style="green")
    for name, value in report.metrics.items():
        table.add_row(name, _fmt(value))
    console.print(table)

    manifest = build_manifest(
        experiment=cfg.name,
        config_hash=cfg.config_hash(),
        corpus_id=cfg.corpus,
        index_id=pipeline.index_id,
        seed=cfg.seed,
        model_versions={"embedder": pipeline.embedder.model_id},
        metrics=report.metrics,
    )
    run_dir = JSONRunTracker().log(manifest, report)
    console.print(f"logged run -> [blue]{run_dir}[/]")

    if smoke:
        hcr = report.metrics.get("hallucinated_citation_rate", 1.0)
        if hcr != 0.0:
            console.print(f"[red]SMOKE FAIL: hallucinated_citation_rate={hcr} (must be 0)[/]")
            raise typer.Exit(code=1)
        console.print("[green]SMOKE OK[/] — pipeline ran end-to-end; zero hallucinated citations")


@app.command()
def graph(
    doc_id: str,
    experiment: str = typer.Option("e7_graph", "--experiment", "-e"),
) -> None:
    """Show a document's citation-graph position: authority, treatment, neighbours."""
    cfg = load_experiment(experiment)
    corpus = load_corpus(cfg.corpus)
    from legalrag.core.graph.citation_graph import CitationGraph

    g = CitationGraph.from_corpus(corpus)
    if doc_id not in corpus:
        console.print(f"[red]{doc_id!r} not in corpus[/]")
        raise typer.Exit(code=1)
    auth = g.authority_scores().get(doc_id, 0.0)
    console.print(f"[cyan]{doc_id}[/]  authority={auth:.3f}")
    status = "OVERRULED" if g.is_overruled(doc_id) else ("questioned" if g.is_questioned(doc_id) else "good law")
    colour = "red" if g.is_questioned(doc_id) else "green"
    console.print(f"status: [{colour}]{status}[/]  incoming treatments: "
                  f"{', '.join(t.value for t in g.incoming_treatments(doc_id)) or '—'}")
    console.print(f"cites ({len(g.cites(doc_id))}): {', '.join(g.cites(doc_id)) or '—'}")
    console.print(f"cited_by ({len(g.cited_by(doc_id))}): {', '.join(g.cited_by(doc_id)) or '—'}")


@app.command()
def ingest(source: str) -> None:
    """Placeholder — jurisdiction packs implement connectors (US/PK tracks)."""
    console.print(f"[yellow]ingest[/] for {source!r} is implemented in jurisdiction packs.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
