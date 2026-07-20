"""Local OpenAI-compatible chat client — the data-sovereignty generation path.

Talks the ``/chat/completions`` dialect that Ollama, vLLM, and LM Studio all
serve, so the same client drives any of them by pointing ``base_url`` at the local
server. No text leaves the machine (ADR 0005). ``httpx`` is imported lazily so this
module registers/imports with zero heavy deps and the smoke path stays hermetic.

Endpoints (all local): Ollama ``http://localhost:11434/v1`` (default),
vLLM ``http://localhost:8000/v1``, LM Studio ``http://localhost:1234/v1``.
Override with ``LOCAL_LLM_BASE_URL``. Install: ``pip install -e ".[llm]"``.
"""

from __future__ import annotations

import os

_DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama


class OpenAICompatClient:
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = (
            base_url or os.environ.get("LOCAL_LLM_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        # Local servers ignore the key but the OpenAI wire format wants one present.
        self._api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return f"local:{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "'httpx' is required for the local LLM client. Install with "
                '`pip install -e ".[llm]"`.'
            ) from e
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])
