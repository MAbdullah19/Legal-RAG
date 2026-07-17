"""Citation graph + authority scoring (master plan §5.5).

The graph is *native* to the legal domain — its edges are real citations that
ingestion resolved into ``citations_out`` — not an LLM-synthesized graph
(contrast GraphRAG, ADR 0004). It is the substrate for authority-weighted
retrieval, "is this still good law?" signals, and precedent-chain (multi-hop)
recall.
"""
