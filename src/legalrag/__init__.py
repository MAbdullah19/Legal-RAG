"""Legal-RAG: research platform for advanced RAG over judicial corpora.

The public surface is intentionally small. Everything the pipeline runs is
resolved by name through :mod:`legalrag.core.registry`, so new techniques are
added as registered components and selected in experiment configs — never by
editing the pipeline runner.
"""

__version__ = "0.1.0"
