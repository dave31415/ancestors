"""Process-local session holding the loaded GEDCOM.

The session is a deliberate seam between data loading (slow, file-IO bound)
and tool execution (hot, must be fast and cheap to invoke). The DSL tools
look up the bound database via `current_session()` rather than receiving it
as an argument — this is what lets the agent-facing tool signatures be
strictly id-based and JSON-friendly.

For now this is a process-wide singleton. The same shape generalizes to:
- per-request bindings in a web context (bind/unbind on request boundaries)
- per-test bindings (pytest fixture binds for the test scope)
- multi-tree comparisons (a richer Session holding multiple registries)

None of those are needed yet. Singleton is the simplest correct thing now.
"""

from __future__ import annotations

from dataclasses import dataclass

from ancestors.models import GedcomDatabase


@dataclass
class Session:
    db: GedcomDatabase


_current: Session | None = None


class SessionNotBoundError(RuntimeError):
    """Raised when a tool runs without an active session."""


def bind_session(db: GedcomDatabase) -> Session:
    """Install a database as the active session and return it."""
    global _current
    _current = Session(db=db)
    return _current


def clear_session() -> None:
    """Detach the active session. Tools that run after this will error."""
    global _current
    _current = None


def current_session() -> Session:
    if _current is None:
        raise SessionNotBoundError(
            "No active session. Call bind_session(load_gedcom(...)) first."
        )
    return _current
