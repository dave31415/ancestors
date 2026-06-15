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

import sqlite3
from dataclasses import dataclass, field

from ancestors.models import GedcomDatabase


@dataclass
class Session:
    db: GedcomDatabase
    # Hardened in-memory SQLite mirror of `db`. Populated by bind_session()
    # so the run_sql tool has somewhere to dispatch. The in-memory
    # GedcomDatabase remains authoritative for the typed-tool surface; this
    # is a parallel read-only view, not a replacement.
    sqlite_connection: sqlite3.Connection | None = field(default=None)


_current: Session | None = None


class SessionNotBoundError(RuntimeError):
    """Raised when a tool runs without an active session."""


def bind_session(
    db: GedcomDatabase, *, with_sqlite: bool = True
) -> Session:
    """Install a database as the active session and return it.

    Building the SQLite mirror takes a few tens of milliseconds and isn't
    needed by every code path (e.g. the focused unit tests in test_gedcom),
    so it can be turned off with with_sqlite=False.
    """
    global _current
    if with_sqlite:
        # Local import to keep session.py free of a hard dep on the SQL store
        # — relevant for tests that don't exercise SQL.
        from ancestors.sqlite_store import build_sqlite_for_session

        sqlite_conn = build_sqlite_for_session(db)
    else:
        sqlite_conn = None
    _current = Session(db=db, sqlite_connection=sqlite_conn)
    return _current


def clear_session() -> None:
    """Detach the active session. Tools that run after this will error."""
    global _current
    if _current is not None and _current.sqlite_connection is not None:
        from ancestors.sqlite_store.safe_conn import forget_timeout

        forget_timeout(_current.sqlite_connection)
        _current.sqlite_connection.close()
    _current = None


def current_session() -> Session:
    if _current is None:
        raise SessionNotBoundError(
            "No active session. Call bind_session(load_gedcom(...)) first."
        )
    return _current
