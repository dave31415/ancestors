"""SQLite schema for the GEDCOM corpus.

Design notes:

- The `individuals` table is wide and denormalises birth/death/burial events
  because nearly every genealogical query touches these. Denormalisation
  trades NULLs (when data is missing) for JOIN-free common-case queries.

- `families` similarly denormalises the marriage event.

- The `parent_child` table is a materialised graph edge. Recursive ancestor
  and descendant CTEs walk this once with one index hit per generation.
  Without it, every recursion step would JOIN family_children to families.

- `other_events` covers everything not denormalised above (OCCU, RESI, BAPM,
  IMMI, EMIG, CENS). The polymorphism here is bounded — all events share a
  common (year, date_raw, place) shape, so one table with a discriminator
  works cleanly.

- Notes get their own multi-valued side table; some Geni notes are paragraphs
  and would inflate the wide individuals row.

- Source citations are coarse-grained for now: per-event `*_has_source`
  boolean on the wide row. A `source_citations` table would be the next
  step if we ever need to introspect individual citation strings.
"""

from __future__ import annotations

import sqlite3

CREATE_STATEMENTS = [
    # ---- individuals: wide denormalised row -------------------------------
    """
    CREATE TABLE individuals (
        id TEXT PRIMARY KEY,
        primary_name TEXT,
        given TEXT,
        surname TEXT,
        sex TEXT,

        birth_year INTEGER,
        birth_date_raw TEXT,
        birth_is_approximate INTEGER,
        birth_country TEXT,
        birth_state TEXT,
        birth_city TEXT,
        birth_place_raw TEXT,
        birth_has_source INTEGER,

        death_year INTEGER,
        death_date_raw TEXT,
        death_is_approximate INTEGER,
        death_country TEXT,
        death_state TEXT,
        death_city TEXT,
        death_place_raw TEXT,
        death_has_source INTEGER,

        burial_year INTEGER,
        burial_date_raw TEXT,
        burial_country TEXT,
        burial_state TEXT,
        burial_city TEXT
    )
    """,
    "CREATE INDEX idx_ind_surname ON individuals(surname)",
    "CREATE INDEX idx_ind_given ON individuals(given)",
    "CREATE INDEX idx_ind_birth_year ON individuals(birth_year)",
    "CREATE INDEX idx_ind_death_year ON individuals(death_year)",
    "CREATE INDEX idx_ind_birth_country ON individuals(birth_country)",
    "CREATE INDEX idx_ind_sex ON individuals(sex)",
    # ---- families ---------------------------------------------------------
    """
    CREATE TABLE families (
        id TEXT PRIMARY KEY,
        husband_id TEXT REFERENCES individuals(id),
        wife_id TEXT REFERENCES individuals(id),
        marriage_year INTEGER,
        marriage_date_raw TEXT,
        marriage_country TEXT,
        marriage_state TEXT,
        marriage_city TEXT
    )
    """,
    "CREATE INDEX idx_fam_husband ON families(husband_id)",
    "CREATE INDEX idx_fam_wife ON families(wife_id)",
    "CREATE INDEX idx_fam_marriage_year ON families(marriage_year)",
    # ---- family_children: explicit children edge --------------------------
    """
    CREATE TABLE family_children (
        family_id TEXT REFERENCES families(id),
        child_id TEXT REFERENCES individuals(id),
        PRIMARY KEY (family_id, child_id)
    )
    """,
    "CREATE INDEX idx_fc_child ON family_children(child_id)",
    # ---- parent_child: materialised graph edge ---------------------------
    """
    CREATE TABLE parent_child (
        parent_id TEXT REFERENCES individuals(id),
        child_id TEXT REFERENCES individuals(id),
        parent_sex TEXT,
        PRIMARY KEY (parent_id, child_id)
    )
    """,
    "CREATE INDEX idx_pc_child ON parent_child(child_id)",
    "CREATE INDEX idx_pc_parent ON parent_child(parent_id)",
    # ---- other_events: non-denormalised event types -----------------------
    """
    CREATE TABLE other_events (
        individual_id TEXT REFERENCES individuals(id),
        event_type TEXT,
        year INTEGER,
        date_raw TEXT,
        country TEXT,
        state TEXT,
        city TEXT,
        place_raw TEXT
    )
    """,
    "CREATE INDEX idx_oe_ind ON other_events(individual_id)",
    "CREATE INDEX idx_oe_type ON other_events(event_type)",
    # ---- individual_notes: multi-valued text ------------------------------
    """
    CREATE TABLE individual_notes (
        individual_id TEXT REFERENCES individuals(id),
        note_index INTEGER,
        text TEXT,
        PRIMARY KEY (individual_id, note_index)
    )
    """,
    "CREATE INDEX idx_notes_ind ON individual_notes(individual_id)",
]


def apply(conn: sqlite3.Connection) -> None:
    """Apply the schema to a fresh connection."""
    cur = conn.cursor()
    for stmt in CREATE_STATEMENTS:
        cur.execute(stmt)
    conn.commit()
