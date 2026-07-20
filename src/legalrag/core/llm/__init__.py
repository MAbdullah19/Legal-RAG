"""LLM client abstraction (master plan §3.2 principle 4, §5.6).

One ``LLMClient`` interface; implementations for local OpenAI-compatible serving
(Ollama / vLLM / LM Studio) and, later, hosted APIs. Under the local-first posture
(ADR 0005) the default generation path never leaves the machine, so confidential
corpora can be answered over without any text crossing the network.
"""

from __future__ import annotations

from legalrag.core.llm.base import LLMClient
from legalrag.core.llm.local import OpenAICompatClient

__all__ = ["LLMClient", "OpenAICompatClient"]
