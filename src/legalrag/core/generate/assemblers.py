"""Context assembly (master plan §5.6).

``concat`` tags each source ``[S1]``, ``[S2]``… and packs chunk text under a
token budget — enough for the E0 baseline. The parent-paragraph (small-to-big)
assembler that expands retrieved chunks to their enclosing section is the E3
contender and registers here later.
"""

from __future__ import annotations

from collections.abc import Sequence

from legalrag.core.interfaces import AssembledContext, Stage
from legalrag.core.models import ScoredChunk
from legalrag.core.registry import register
from legalrag.core.text import estimate_tokens


@register(Stage.ASSEMBLER, "concat")
class ConcatAssembler:
    def __init__(self, budget_tokens: int = 8000) -> None:
        self.budget_tokens = budget_tokens

    def assemble(self, query: str, chunks: Sequence[ScoredChunk]) -> AssembledContext:
        used: list[ScoredChunk] = []
        lines: list[str] = []
        spent = 0
        for sc in chunks:
            t = estimate_tokens(sc.text)
            if used and spent + t > self.budget_tokens:
                break
            tag = f"S{len(used) + 1}"
            pin = f"{sc.doc_id} ¶ {','.join(sc.para_ids)}"
            lines.append(f"[{tag}] ({pin})\n{sc.text}")
            used.append(sc)
            spent += t
        return AssembledContext(text="\n\n".join(lines), chunks=used)
