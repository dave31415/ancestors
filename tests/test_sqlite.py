"""Tests for the SQLite escape hatch.

Three layers covered:

- Schema + loader: tables exist, row counts match the in-memory db,
  parent_child edges are populated for both parents.
- Safety wrappers: query_only blocks writes, ATTACH is denied, timeout
  fires on a deliberately slow CTE, result rows are capped.
- run_sql tool surface: validates query length, returns dicts, surfaces
  SQL errors as ToolErrors via the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.dispatch import Dispatcher
from ancestors.session import bind_session, clear_session, current_session
from ancestors.sqlite_store import MAX_ROWS, build_sqlite_for_session
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"


@pytest.fixture(scope="module")
def db():
    return load_gedcom(GEDCOM_PATH)


@pytest.fixture(scope="module", autouse=True)
def session(db):
    bind_session(db)
    yield
    clear_session()


@pytest.fixture
def disp():
    return Dispatcher()


# ---------------------------------------------------------------------------
# Schema + loader
# ---------------------------------------------------------------------------


def test_individuals_table_row_count(db):
    conn = current_session().sqlite_connection
    n = conn.execute("SELECT COUNT(*) FROM individuals").fetchone()[0]
    assert n == db.individual_count


def test_families_table_row_count(db):
    conn = current_session().sqlite_connection
    n = conn.execute("SELECT COUNT(*) FROM families").fetchone()[0]
    assert n == db.family_count


def test_parent_child_includes_both_parents():
    conn = current_session().sqlite_connection
    # David has both parents recorded; we expect two parent_child rows.
    n = conn.execute(
        "SELECT COUNT(*) FROM parent_child WHERE child_id = ?",
        (DAVID_ID,),
    ).fetchone()[0]
    assert n == 2
    sexes = {
        row[0]
        for row in conn.execute(
            "SELECT parent_sex FROM parent_child WHERE child_id = ?",
            (DAVID_ID,),
        )
    }
    assert sexes == {"M", "F"}


def test_individual_columns_for_known_record():
    conn = current_session().sqlite_connection
    row = conn.execute(
        "SELECT primary_name, surname, sex, birth_year, birth_country "
        "FROM individuals WHERE id = ?",
        (DAVID_ID,),
    ).fetchone()
    assert row["surname"] == "Johnston"
    assert row["sex"] == "M"
    assert row["birth_year"] == 1975
    assert row["birth_country"] == "United States"


def test_other_events_carry_event_type():
    conn = current_session().sqlite_connection
    types = {
        row[0]
        for row in conn.execute("SELECT DISTINCT event_type FROM other_events")
    }
    # Geni exports tend to carry occupation/residence/baptism etc.
    assert types  # at least some rows present


# ---------------------------------------------------------------------------
# Safety wrappers
# ---------------------------------------------------------------------------


def test_query_only_blocks_writes(disp):
    result = disp.dispatch(
        "run_sql", {"query": "INSERT INTO individuals (id) VALUES ('hack')"}
    )
    assert result.ok is False
    assert "tool_exception" in result.error.code or "sql" in result.error.message.lower()


def test_attach_database_denied(disp):
    result = disp.dispatch(
        "run_sql",
        {"query": "ATTACH DATABASE '/tmp/evil.db' AS evil"},
    )
    assert result.ok is False


def test_timeout_fires_on_runaway_cte(db):
    # A recursive CTE with no termination beyond a large bound — should be
    # killed by the progress handler well before completing.
    from ancestors.sqlite_store.safe_conn import (
        QUERY_TIMEOUT_SECONDS,
        _install_timeout,
    )
    import sqlite3

    conn = current_session().sqlite_connection
    # Tighten the deadline so the test runs fast.
    _install_timeout(conn, 0.5)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                """
                WITH RECURSIVE bomb(n) AS (
                    SELECT 1
                    UNION ALL
                    SELECT n + 1 FROM bomb WHERE n < 100000000
                )
                SELECT COUNT(*) FROM bomb;
                """
            ).fetchall()
    finally:
        _install_timeout(conn, QUERY_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# run_sql tool surface
# ---------------------------------------------------------------------------


def test_run_sql_returns_dicts(disp):
    result = disp.dispatch(
        "run_sql",
        {"query": "SELECT id, surname, birth_year FROM individuals WHERE birth_year = 1975"},
    )
    assert result.ok
    assert isinstance(result.value, list)
    assert len(result.value) >= 1
    assert "surname" in result.value[0]


def test_run_sql_rejects_empty_query(disp):
    result = disp.dispatch("run_sql", {"query": ""})
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_run_sql_rejects_overlong_query(disp):
    result = disp.dispatch("run_sql", {"query": "SELECT 1; " + "-" * 6000})
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_recursive_cte_for_ancestors(disp):
    # The canonical ancestor walk — this is the query the agent should
    # generate for "ancestors of X to depth N".
    q = f"""
    WITH RECURSIVE anc(person_id, ancestor_id, gen) AS (
      SELECT '{DAVID_ID}', '{DAVID_ID}', 0
      UNION ALL
      SELECT a.person_id, pc.parent_id, a.gen + 1
      FROM anc a JOIN parent_child pc ON a.ancestor_id = pc.child_id
      WHERE a.gen < 3
    )
    SELECT a.gen, i.primary_name, i.birth_year
    FROM anc a JOIN individuals i ON i.id = a.ancestor_id
    ORDER BY a.gen, i.primary_name;
    """
    result = disp.dispatch("run_sql", {"query": q})
    assert result.ok
    # David at gen=0, parents at gen=1, grandparents at gen=2,
    # great-grandparents at gen=3. With perfect coverage that's 15 rows.
    assert 1 <= len(result.value) <= MAX_ROWS
    gens = {row["gen"] for row in result.value}
    assert 0 in gens
    assert 3 in gens


def test_cousin_marriage_query_runs(disp):
    """The motivating use case: a single CTE answering the cousin question.

    We don't assert specific consanguineous marriages exist — we assert the
    SQL is syntactically valid and runs to completion without timing out.
    """
    q = """
    WITH RECURSIVE anc(person_id, ancestor_id, gen) AS (
      SELECT id, id, 0 FROM individuals
      UNION ALL
      SELECT a.person_id, pc.parent_id, a.gen + 1
      FROM anc a JOIN parent_child pc ON a.ancestor_id = pc.child_id
      WHERE a.gen < 3
    )
    SELECT
      f.id AS family_id,
      ha.ancestor_id AS shared,
      ha.gen + wa.gen AS kinship_total
    FROM families f
    JOIN anc ha ON ha.person_id = f.husband_id
    JOIN anc wa ON wa.person_id = f.wife_id AND wa.ancestor_id = ha.ancestor_id
    WHERE ha.gen + wa.gen > 0
    ORDER BY kinship_total;
    """
    result = disp.dispatch("run_sql", {"query": q})
    assert result.ok
    # The query is well-formed; either we found cousin marriages or we didn't.
    assert isinstance(result.value, list)
