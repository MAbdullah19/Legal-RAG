"""FastAPI adapter — HTTP front door over :class:`LegalRAGService`.

For general callers (web UIs, notebooks, scripts, other backends) and for agents
whose framework speaks HTTP rather than MCP. Thin by design: every route calls
the neutral core and returns its shared response models.

Run::

    LEGALRAG_EXPERIMENT=e0_naive_baseline python -m legalrag.service.api
    # or, with uvicorn's factory flag (env selects the experiment):
    uvicorn --factory legalrag.service.api:create_app

Requires the ``serve`` extra: ``pip install -e ".[serve]"``.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from legalrag.service.core import (
    DEFAULT_EXPERIMENT,
    AnswerResponse,
    CaseNode,
    LegalRAGService,
    SearchResponse,
)

EXPERIMENT_ENV = "LEGALRAG_EXPERIMENT"


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class HealthResponse(BaseModel):
    status: str
    experiment: str
    index_id: str | None
    docs: int


def create_app(experiment: str | None = None):  # type: ignore[no-untyped-def]
    """Build a FastAPI app whose routes serve one loaded experiment.

    ``experiment`` falls back to ``$LEGALRAG_EXPERIMENT`` then the default. The
    pipeline is loaded (and indexed) once, here at construction time.
    """
    from fastapi import FastAPI, HTTPException

    exp = experiment or os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
    service = LegalRAGService.load(exp)

    app = FastAPI(
        title="Legal-RAG",
        version="0.1.0",
        summary="Grounded, citation-verified RAG over judicial corpora.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            experiment=service.experiment,
            index_id=service.pipeline.index_id,
            docs=len(service.pipeline.corpus),
        )

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        return service.answer(req.query)

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        return service.search(req.query, k=req.k)

    @app.get("/case/{doc_id:path}", response_model=CaseNode)
    def case(doc_id: str) -> CaseNode:
        node = service.case(doc_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"{doc_id!r} not in corpus")
        return node

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("LEGALRAG_HOST", "127.0.0.1")
    port = int(os.environ.get("LEGALRAG_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
