"""The Corpus protocol — the domain's side of the eval boundary.

A Corpus is everything domain-specific that the eval runner needs:
  - a way to make itself the active session (so tools can find data),
  - the system prompt (which encodes the domain's role and reasoning),
  - the typed tool definitions (which encode the domain's action space).

The runner composes a Corpus with a Case and the agent loop. The runner
itself imports nothing from the domain — swap GedcomCorpus for a
MedicalRecordsCorpus and the same runner evaluates the same way.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class Corpus(Protocol):
    """A domain's data + prompt + tools, packaged for the eval runner."""

    name: str

    def bind(self) -> AbstractContextManager[None]:
        """Make this corpus the active session for the enclosed block.

        Implementations call whatever domain-specific session-binding
        machinery they need. The yielded context manager must tear down
        any acquired resources on exit, even when the runner raises.
        """
        ...

    def system_prompt(self) -> str:
        """The fully-rendered system prompt for this corpus."""
        ...

    def tool_defs(self) -> list[dict[str, Any]]:
        """Anthropic tool definitions for the agent's tool surface."""
        ...
