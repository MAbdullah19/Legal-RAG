"""Query transformers (ablation axis C, master plan §5.4). E0 uses ``none``;
intent routing, self-query filters, multi-query and HyDE register here (E6)."""

from __future__ import annotations

from legalrag.core.interfaces import QueryPlan, Stage
from legalrag.core.registry import register


@register(Stage.QUERY_TRANSFORMER, "none")
class IdentityQueryTransformer:
    def transform(self, query: str) -> QueryPlan:
        return QueryPlan(queries=[query])
