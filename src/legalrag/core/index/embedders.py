"""Embedders (ablation axis B, master plan §5.3).

``hashing`` is a deterministic, dependency-free embedder so the E0 smoke path
needs no model downloads or API keys. It maps tokens into a fixed-dim space by
hashing with log-tf weighting and L2 normalization — a bag-of-words vector whose
cosine similarity tracks lexical overlap. It is a *baseline stand-in*, not a
semantic model: voyage/BGE-M3/OpenAI embedders register here and are selected in
config for real experiments (E5 bake-off).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from legalrag.core.interfaces import Stage
from legalrag.core.registry import register
from legalrag.core.text import tokenize


@register(Stage.EMBEDDER, "hashing")
class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % self._dim
        sign = 1.0 if h[4] & 1 else -1.0  # signed hashing reduces collisions
        return idx, sign

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, c in counts.items():
            idx, sign = self._bucket(tok)
            vec[idx] += sign * (1.0 + math.log(c))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
