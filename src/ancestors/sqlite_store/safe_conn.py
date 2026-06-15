"""Hardened SQLite connection — the security boundary for run_sql.

Defences, in priority order:

1. **Read-only by construction.** `PRAGMA query_only = ON` rejects INSERT,
   UPDATE, DELETE, and (importantly) any temp-table writes. The schema is
   applied and populated *before* this pragma is set, so loading still
   works.

2. **No filesystem reach.** `enable_load_extension(False)` plus an
   authorizer callback that denies ATTACH/DETACH prevent the model from
   pivoting into other SQLite files or loading shared libraries.

3. **Wall-clock timeout.** A progress handler checks `time.time()` every
   N opcodes and returns non-zero (which SQLite interprets as "abort")
   once a deadline is exceeded.

4. **Result row cap.** Enforced at the run_sql callsite via fetchmany.

5. **Query length cap.** Enforced by the dispatcher's args validation
   (BoundedStr at MAX_QUERY_LEN chars).

The connection is in-memory and per-session — there is no file on disk,
no shared connection, no cross-session contamination.
"""

from __future__ import annotations

import sqlite3
import time

from ancestors.models import GedcomDatabase
from ancestors.sqlite_store import loader, schema

# Public knobs the dispatcher/wrapper read.
QUERY_TIMEOUT_SECONDS = 10.0
MAX_ROWS = 1000

# Progress handler granularity. SQLite calls back every N VDBE opcodes;
# 1_000_000 is roughly tens of milliseconds and keeps the overhead negligible.
_PROGRESS_GRANULARITY = 1_000_000

# Per-connection state for the wall-clock timeout. Keyed by id(conn) because
# sqlite3.Connection objects don't support custom attributes or weak refs.
# Entries are popped at clear_session() time.
_timeout_state: dict[int, dict] = {}


def build_sqlite_for_session(db: GedcomDatabase) -> sqlite3.Connection:
    """Return a hardened, populated, read-only connection for one session."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Populate first; lock down second.
    schema.apply(conn)
    loader.populate(conn, db)

    # ---- Lock-down phase ----
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA cell_size_check = ON")
    try:
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.NotSupportedError):
        # Some builds disable enable_load_extension entirely — that's fine,
        # it just means there's nothing to lock further.
        pass
    conn.set_authorizer(_authorizer)
    _install_timeout(conn, QUERY_TIMEOUT_SECONDS)

    return conn


def _authorizer(action: int, *_args) -> int:
    """SQLite authorizer callback — denies dangerous actions."""
    # ATTACH/DETACH would let the model pivot into other SQLite files.
    if action == sqlite3.SQLITE_ATTACH:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_DETACH:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _install_timeout(conn: sqlite3.Connection, seconds: float) -> None:
    """Install a wall-clock timeout via set_progress_handler.

    The handler reads a mutable deadline stored in `_timeout_state`; run_sql
    resets that deadline before each query. Returning non-zero aborts the
    in-progress statement.
    """
    state = {"deadline": time.time() + seconds, "limit_seconds": seconds}

    def _check() -> int:
        return 1 if time.time() > state["deadline"] else 0

    conn.set_progress_handler(_check, _PROGRESS_GRANULARITY)
    _timeout_state[id(conn)] = state


def reset_timeout(conn: sqlite3.Connection) -> None:
    """Restart the deadline clock — called by run_sql before each query."""
    state = _timeout_state.get(id(conn))
    if state is not None:
        state["deadline"] = time.time() + state["limit_seconds"]


def forget_timeout(conn: sqlite3.Connection) -> None:
    """Drop the timeout state for a connection. Called when the session ends."""
    _timeout_state.pop(id(conn), None)
