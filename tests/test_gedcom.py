"""Smoke tests for the GEDCOM tool layer against the real export file.

These tests are intentionally evidence-anchored: they assert against specific
records the researcher curated (David Johnston as the proband, Peter Gallagher
as a working-hypothesis ancestor, Dr. Lancelot Johnston as the DAR/SAR anchor).
If counts or anchor facts drift, that should fail loudly rather than silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.models import EventType, Sex
from ancestors.tools.gedcom import (
    GedcomLoadError,
    IndividualNotFoundError,
    get_individual,
    load_gedcom,
)

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"


@pytest.fixture(scope="module")
def db():
    return load_gedcom(GEDCOM_PATH)


def test_load_counts(db):
    assert db.individual_count == 336
    assert db.family_count == 230


def test_load_missing_file_raises():
    with pytest.raises(GedcomLoadError):
        load_gedcom("/tmp/does-not-exist.ged")


def test_get_individual_unknown_id(db):
    with pytest.raises(IndividualNotFoundError):
        get_individual(db, "@I_NOPE@")


def test_proband_david_johnston(db):
    david = get_individual(db, "@I6000000001904015159@")
    assert david.sex == Sex.MALE
    assert david.primary_name is not None
    assert david.primary_name.surname == "Johnston"
    assert "David" in (david.primary_name.given or "")
    assert david.birth is not None
    assert david.birth.date is not None
    assert david.birth.date.year == 1975
    assert david.families_as_child == ["@F4025896807070010511@"]


def test_peter_gallagher_birth_is_approximate(db):
    peter = get_individual(db, "@I1126081@") if "@I1126081@" in db.individuals else None
    # The Peter Gallagher xref isn't a stable assumption — search by name instead.
    if peter is None:
        peter = next(
            ind
            for ind in db.individuals.values()
            if (n := ind.primary_name) is not None
            and n.surname == "Gallagher"
            and (n.given or "").strip().lower().startswith("peter")
            and ind.birth is not None
            and ind.birth.date is not None
            and ind.birth.date.year is not None
            and 1790 <= ind.birth.date.year <= 1810
        )
    assert peter.birth is not None
    assert peter.birth.date is not None
    assert peter.birth.date.is_approximate is True
    assert peter.birth.date.year is not None
    # Spec says ~1798; the export here records ABT 1797. Both consistent.
    assert 1795 <= peter.birth.date.year <= 1800


def test_lancelot_johnston_present(db):
    candidates = [
        ind
        for ind in db.individuals.values()
        if (n := ind.primary_name) is not None
        and n.surname == "Johnston"
        and "Lancelot" in (n.given or "")
    ]
    assert candidates, "expected at least one Lancelot Johnston in the export"


def test_family_links_resolve(db):
    david = get_individual(db, "@I6000000001904015159@")
    fam_id = david.families_as_child[0]
    fam = db.families[fam_id]
    assert david.id in fam.children_ids
    assert fam.husband_id is not None
    assert fam.wife_id is not None


def test_event_types_round_trip(db):
    # Every individual with a birth event has type BIRTH.
    with_birth = [ind for ind in db.individuals.values() if ind.birth is not None]
    assert with_birth, "expected at least some birth events"
    assert all(ind.birth.type == EventType.BIRTH for ind in with_birth)
