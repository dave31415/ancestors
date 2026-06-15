"""Populate the SQLite schema from an in-memory GedcomDatabase.

Mechanical Pydantic-to-tuple conversion. The result is a snapshot of the
GEDCOM that supports SQL queries — the in-memory `GedcomDatabase` remains
authoritative for the typed-tool surface; SQLite is a parallel view.
"""

from __future__ import annotations

import sqlite3

from ancestors.models import Event, Family, GedcomDatabase, Individual, Place


def populate(conn: sqlite3.Connection, db: GedcomDatabase) -> None:
    """Insert every row from the in-memory db into the connection's tables.

    Assumes the schema is already applied. Idempotent on a fresh connection
    only — re-running on an already-populated DB will fail PRIMARY KEY
    constraints, which is the desired behaviour (it would mean a bug).
    """
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT INTO individuals VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """,
        (_individual_row(i) for i in db.individuals.values()),
    )

    cur.executemany(
        "INSERT INTO families VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_family_row(f) for f in db.families.values()),
    )

    # family_children + parent_child are built from family.children_ids.
    cur.executemany(
        "INSERT INTO family_children VALUES (?, ?)",
        _family_children_rows(db),
    )
    cur.executemany(
        "INSERT OR IGNORE INTO parent_child VALUES (?, ?, ?)",
        _parent_child_rows(db),
    )

    cur.executemany(
        "INSERT INTO other_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        _other_events_rows(db),
    )

    cur.executemany(
        "INSERT INTO individual_notes VALUES (?, ?, ?)",
        _note_rows(db),
    )

    conn.commit()


# ---------------------------------------------------------------------------
# Per-table row generators
# ---------------------------------------------------------------------------


def _individual_row(ind: Individual) -> tuple:
    name = ind.primary_name
    primary = name.full if name else None
    given = name.given if name else None
    surname = name.surname if name else None

    birth = _event_cols(ind.birth, with_source=True)
    death = _event_cols(ind.death, with_source=True)
    burial = _event_cols(ind.burial, with_source=False, with_approx=False)

    return (
        ind.id,
        primary,
        given,
        surname,
        ind.sex.value,
        *birth,
        *death,
        *burial,
    )


def _event_cols(
    event: Event | None,
    *,
    with_source: bool,
    with_approx: bool = True,
) -> tuple:
    """Render a single event into the column tuple it occupies on individuals.

    The column shape depends on which event we're packing:
    - birth/death (with_source=True, with_approx=True): 8 columns
    - burial (with_source=False, with_approx=False): 5 columns
    """
    if event is None:
        if with_source and with_approx:
            return (None, None, None, None, None, None, None, None)
        return (None, None, None, None, None)

    year = event.date.year if event.date else None
    date_raw = event.date.raw if event.date else None
    approx = (
        int(event.date.is_approximate) if event.date and with_approx else None
    )
    place = event.place or Place()
    has_source = int(bool(event.source_ids)) if with_source else None

    if with_source and with_approx:
        return (
            year, date_raw, approx,
            place.country, place.state, place.city, place.raw,
            has_source,
        )
    # burial: no approx flag, no source bool
    return (
        year, date_raw,
        place.country, place.state, place.city,
    )


def _family_row(fam: Family) -> tuple:
    marriage = fam.marriage
    if marriage:
        m_year = marriage.date.year if marriage.date else None
        m_date_raw = marriage.date.raw if marriage.date else None
        m_place = marriage.place or Place()
    else:
        m_year = m_date_raw = None
        m_place = Place()
    return (
        fam.id,
        fam.husband_id,
        fam.wife_id,
        m_year,
        m_date_raw,
        m_place.country,
        m_place.state,
        m_place.city,
    )


def _family_children_rows(db: GedcomDatabase):
    for fam in db.families.values():
        for child_id in fam.children_ids:
            if child_id in db.individuals:
                yield (fam.id, child_id)


def _parent_child_rows(db: GedcomDatabase):
    for fam in db.families.values():
        for child_id in fam.children_ids:
            if child_id not in db.individuals:
                continue
            for parent_id, parent_sex in (
                (fam.husband_id, "M"),
                (fam.wife_id, "F"),
            ):
                if parent_id and parent_id in db.individuals:
                    yield (parent_id, child_id, parent_sex)


def _other_events_rows(db: GedcomDatabase):
    for ind in db.individuals.values():
        for ev in ind.other_events:
            place = ev.place or Place()
            year = ev.date.year if ev.date else None
            date_raw = ev.date.raw if ev.date else None
            yield (
                ind.id,
                ev.type.value,
                year,
                date_raw,
                place.country,
                place.state,
                place.city,
                place.raw,
            )


def _note_rows(db: GedcomDatabase):
    for ind in db.individuals.values():
        for i, note in enumerate(ind.notes):
            yield (ind.id, i, note)
