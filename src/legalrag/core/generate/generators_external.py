"""LLM-backed grounded generator (master plan §5.6, rung E9).

``grounded`` composes an answer with an LLM but stays citation-faithful by
construction downstream: it emits ``AnswerCitation`` rows (doc/paragraph + quote)
that the deterministic verifier then checks against the corpus, dropping anything
fabricated. So even a small local model cannot smuggle a hallucinated citation
past the trust stack — the model proposes, the verifier disposes (§5.7).

Default backend is the local OpenAI-compatible client (ADR 0005): no document text
leaves the machine. Heavy/optional deps stay lazy; the client is built on first
use (or injected for tests), so registering this component is free.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from legalrag.core.generate.prompts import get_prompt
from legalrag.core.interfaces import AssembledContext, Stage
from legalrag.core.llm.base import LLMClient
from legalrag.core.models import AnswerCitation, GeneratedAnswer, ScoredChunk
from legalrag.core.registry import register

_ABSTAIN_TEXT = "The retrieved sources do not answer this question; no grounded answer is available."


@register(Stage.GENERATOR, "grounded")
class GroundedGenerator:
    def __init__(
        self,
        backend: str = "local",
        model: str = "qwen2.5:7b-instruct",
        base_url: str | None = None,
        prompt: str = "grounded_v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_sources: int = 8,
        timeout: float = 120.0,
        client: LLMClient | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_sources = max_sources
        self.timeout = timeout
        self._client = client
        get_prompt(prompt)  # fail fast on an unknown prompt name

    def _ensure_client(self) -> LLMClient:
        if self._client is None:
            if self.backend == "local":
                from legalrag.core.llm.local import OpenAICompatClient

                self._client = OpenAICompatClient(
                    model=self.model, base_url=self.base_url, timeout=self.timeout
                )
            else:
                raise ValueError(f"unknown generator backend {self.backend!r}")
        return self._client

    @staticmethod
    def _render_sources(chunks: Sequence[ScoredChunk]) -> str:
        blocks = []
        for i, sc in enumerate(chunks, start=1):
            pin = f"{sc.doc_id} ¶ {','.join(sc.para_ids)}" if sc.para_ids else sc.doc_id
            blocks.append(f"[{i}] ({pin})\n{sc.text}")
        return "\n\n".join(blocks)

    def generate(self, query: str, context: AssembledContext) -> GeneratedAnswer:
        chunks = list(context.chunks[: self.max_sources])
        if not chunks:
            return GeneratedAnswer(qid="", text=_ABSTAIN_TEXT, abstained=True)

        system, user_tmpl = get_prompt(self.prompt)
        user = user_tmpl.format(query=query, sources=self._render_sources(chunks))
        raw = self._ensure_client().complete(
            system, user, temperature=self.temperature, max_tokens=self.max_tokens
        )

        parsed = _extract_json(raw)
        if parsed is None:  # fail closed — a legal answer we can't parse is not emitted
            return GeneratedAnswer(qid="", text=_ABSTAIN_TEXT, abstained=True)

        if parsed.get("abstain") or not str(parsed.get("answer", "")).strip():
            return GeneratedAnswer(qid="", text=_ABSTAIN_TEXT, abstained=True)

        citations: list[AnswerCitation] = []
        used: list[str] = []
        for c in parsed.get("citations", []) or []:
            idx = _as_source_index(c.get("source"))
            if idx is None or not (1 <= idx <= len(chunks)):
                continue
            sc = chunks[idx - 1]
            quote = c.get("quote")
            citations.append(
                AnswerCitation(
                    doc_id=sc.doc_id,
                    para_ids=list(sc.para_ids),
                    quote=str(quote) if quote else None,
                )
            )
            used.append(sc.chunk_id)

        return GeneratedAnswer(
            qid="",
            text=str(parsed["answer"]).strip(),
            citations=citations,
            used_chunk_ids=used,
        )


def _as_source_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group())
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a single JSON object from an LLM response (tolerates
    ```json fences and surrounding prose). Returns None if nothing parses."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
