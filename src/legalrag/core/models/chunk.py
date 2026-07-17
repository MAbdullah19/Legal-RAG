"""Chunk: the unit that gets embedded and indexed.

A chunk spans one or more contiguous paragraphs and keeps their pids so
retrieval results map straight back to pinpoint-citable units.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str                       # "{doc_id}#{first_pid}-{last_pid}"
    doc_id: str
    para_ids: list[str]                 # pinpoint-citable units this chunk covers
    text: str
    context_prefix: str | None = None   # contextual-retrieval annotation (E4)
    token_estimate: int | None = None
    payload: dict[str, object] = Field(default_factory=dict)  # court_level, date... -> filters

    def embed_text(self) -> str:
        """Text actually embedded. The context prefix is embedded but never shown
        to the user (Anthropic contextual retrieval, master plan §5.2)."""
        if self.context_prefix:
            return f"{self.context_prefix.strip()}\n\n{self.text}"
        return self.text
