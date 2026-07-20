"""The provider-neutral chat interface every generator talks to.

Thin by design (§3.2 principle 3 — no framework backbone): a single ``complete``
call over a system+user turn returning text. Routing across local/hosted backends
is a config choice, not a code change; generators depend only on this Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...
