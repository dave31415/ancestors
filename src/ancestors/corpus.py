"""GedcomCorpus — genealogy implementation of the eval Corpus protocol.

Wraps a loaded GedcomDatabase together with the session binding, system
prompt builder, and tool schema export. Everything genealogy-specific
lives here; the generic runner uses it through the Corpus protocol.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ancestors.genealogy_prompts import build_system_prompt
from ancestors.agent.schema import export_tools_for_anthropic
from ancestors.models import GedcomDatabase
from ancestors.session import bind_session, clear_session
from ancestors.tools.gedcom import load_gedcom


class GedcomCorpus:
    """Genealogy corpus: a loaded GEDCOM plus its prompt and tool surface."""

    name = "ancestors"

    def __init__(self, db: GedcomDatabase) -> None:
        self.db = db

    @classmethod
    def from_path(cls, gedcom_path: Path) -> "GedcomCorpus":
        return cls(load_gedcom(gedcom_path))

    @contextmanager
    def bind(self) -> Iterator[None]:
        bind_session(self.db)
        try:
            yield
        finally:
            clear_session()

    def system_prompt(self) -> str:
        return build_system_prompt(self.db)

    def tool_defs(self) -> list[dict[str, Any]]:
        return export_tools_for_anthropic()
