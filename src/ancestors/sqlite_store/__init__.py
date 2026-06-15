"""SQLite store for GEDCOM data — the SQL escape hatch.

Built alongside the typed-tool surface, not as a replacement. The agent
gets one new tool (run_sql) and the existing 24 tools continue to work.
We'll observe which surface the agent reaches for on real questions and
decide later whether to deprecate parts of the typed tool inventory.

Module responsibilities:

  schema.py    — table definitions and indexes
  loader.py    — populates SQLite from an in-memory GedcomDatabase
  safe_conn.py — hardened read-only connection (query_only, timeouts,
                 no ATTACH, no load_extension, row cap)

Initialise per session via build_sqlite_for_session(db).
"""

from ancestors.sqlite_store.safe_conn import (
    MAX_ROWS,
    QUERY_TIMEOUT_SECONDS,
    build_sqlite_for_session,
)

__all__ = ["build_sqlite_for_session", "MAX_ROWS", "QUERY_TIMEOUT_SECONDS"]
