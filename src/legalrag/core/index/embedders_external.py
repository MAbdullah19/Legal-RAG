"""Real embedder adapters behind the same Embedder Protocol (master plan §5.3).

These implement the hybrid/provider-abstraction principle: swapping the
deterministic ``hashing`` baseline for a production embedder is a config change.
Heavy/paid dependencies are imported **lazily** so importing this module (and
thus registering the components) costs nothing and the smoke path stays hermetic
(ADR 0003). Instantiating one without its extra installed raises a clear,
actionable error.

Install: ``pip install -e ".[embeddings]"`` · keys via ``.env`` (see .env.example).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from legalrag.core.interfaces import Stage
from legalrag.core.registry import register


def _missing(pkg: str, extra: str) -> ImportError:
    return ImportError(
        f"{pkg!r} is required for this embedder. Install with "
        f'`pip install -e ".[{extra}]"` (kept out of the core/smoke deps by design).'
    )


@register(Stage.EMBEDDER, "sentence_transformer")
class SentenceTransformerEmbedder:
    """Local embeddings (default BGE-M3) — the data-sovereignty path, no API key.
    Downloads the model on first use."""

    def __init__(self, model: str = "BAAI/bge-m3", normalize: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise _missing("sentence-transformers", "embeddings") from e
        self._model_name = model
        self._normalize = normalize
        self._st = SentenceTransformer(model)
        self._dim = int(self._st.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return f"st:{self._model_name}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._st.encode(
            list(texts), normalize_embeddings=self._normalize, convert_to_numpy=True
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@register(Stage.EMBEDDER, "voyage")
class VoyageEmbedder:
    """Voyage AI embeddings (incl. ``voyage-law-2``, legal-tuned) — API, best
    quality on legal text. Reads ``VOYAGE_API_KEY`` from the environment."""

    def __init__(self, model: str = "voyage-3-large", input_type: str = "document") -> None:
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise _missing("voyageai", "embeddings") from e
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is not set (see .env.example).")
        self._model = model
        self._input_type = input_type
        self._client = voyageai.Client()
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    @property
    def model_id(self) -> str:
        return f"voyage:{self._model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._client.embed(list(texts), model=self._model, input_type=self._input_type)
        embeddings: list[list[float]] = result.embeddings
        if self._dim is None and embeddings:
            self._dim = len(embeddings[0])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._model, input_type="query")
        return list(result.embeddings[0])
