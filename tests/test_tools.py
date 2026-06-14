"""Tests for the search, lineage, and gap-detection tools.

Anchored to the curated subjects (David Johnston, Peter Gallagher) so that
if scoring, traversal, or gap detection regresses, real-world cases fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.models import GapType, IndividualQuery, Sex
from ancestors.tools.gaps import find_evidence_gaps
from ancestors.tools.gedcom import (
    FamilyNotFoundError,
    get_family,
    load_gedcom,
)
from ancestors.tools.lineage import (
    count_ancestors,
    count_descendants,
    get_ancestors,
    get_descendants,
)
from ancestors.tools.search import find_individuals

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"
PETER_GALLAGHER_ID = "@I6000000001912136867@"


@pytest.fixture(scope="module")
def db():
    return load_gedcom(GEDCOM_PATH)


# ---------------------------------------------------------------------------
# find_individuals
# ---------------------------------------------------------------------------


def test_find_individuals_empty_query_returns_nothing(db):
    assert find_individuals(db, IndividualQuery()) == []


def test_find_individuals_by_surname(db):
    matches = find_individuals(db, IndividualQuery(surname="Gallagher"))
    assert matches, "expected at least one Gallagher"
    # The top match must be an exact-surname Gallagher.
    top = matches[0].individual
    assert top.primary_name is not None
    assert top.primary_name.surname == "Gallagher"
    assert matches[0].score.name == pytest.approx(1.0)


def test_find_individuals_ranks_exact_name_first(db):
    matches = find_individuals(
        db, IndividualQuery(given="Peter", surname="Gallagher")
    )
    assert matches
    top = matches[0].individual
    assert top.primary_name is not None
    assert top.primary_name.surname == "Gallagher"
    assert "peter" in (top.primary_name.given or "").lower()


def test_find_individuals_year_range_constraint(db):
    # Peter Gallagher is the only Peter Gallagher born ~1797 — narrowing by
    # year should still find him at the top.
    matches = find_individuals(
        db,
        IndividualQuery(
            given="Peter",
            surname="Gallagher",
            birth_year_min=1790,
            birth_year_max=1810,
        ),
    )
    assert matches
    assert matches[0].individual.id == PETER_GALLAGHER_ID
    assert matches[0].score.year == 1.0


def test_find_individuals_sex_constraint(db):
    matches = find_individuals(
        db, IndividualQuery(surname="Johnston", sex=Sex.MALE)
    )
    assert matches
    # All top-scored results should be male (or unknown — see scoring note).
    assert all(m.individual.sex != Sex.FEMALE for m in matches[:5])


def test_find_individuals_respects_limit(db):
    matches = find_individuals(db, IndividualQuery(surname="Johnston", limit=3))
    assert len(matches) <= 3


# ---------------------------------------------------------------------------
# get_family
# ---------------------------------------------------------------------------


def test_get_family_resolves(db):
    fam = get_family(db, "@F4025896807070010511@")
    assert fam.husband_id == "@I1126034@"
    assert fam.wife_id == "@I1126058@"
    assert DAVID_ID in fam.children_ids


def test_get_family_unknown_id_raises(db):
    with pytest.raises(FamilyNotFoundError):
        get_family(db, "@F_DOES_NOT_EXIST@")


# ---------------------------------------------------------------------------
# get_ancestors / get_descendants
# ---------------------------------------------------------------------------


def test_get_ancestors_zero_generations(db):
    tree = get_ancestors(db, DAVID_ID, generations=0)
    assert tree.individual.id == DAVID_ID
    assert tree.generation == 0
    assert tree.father is None
    assert tree.mother is None
    assert count_ancestors(tree) == 1


def test_get_ancestors_two_generations(db):
    tree = get_ancestors(db, DAVID_ID, generations=2)
    assert tree.father is not None
    assert tree.mother is not None
    assert tree.father.generation == 1
    # David's mother's parents should be present at generation 2.
    assert tree.mother.father is not None or tree.mother.mother is not None
    # Reasonable bounds for a 2-generation tree (root + up to 6 ancestors).
    assert 2 <= count_ancestors(tree) <= 7


def test_get_ancestors_negative_generations_raises(db):
    with pytest.raises(ValueError):
        get_ancestors(db, DAVID_ID, generations=-1)


def test_get_descendants_of_peter_gallagher(db):
    tree = get_descendants(db, PETER_GALLAGHER_ID, generations=3)
    assert tree.individual.id == PETER_GALLAGHER_ID
    # Peter Gallagher should have at least some descendants in this export.
    assert count_descendants(tree) > 1


# ---------------------------------------------------------------------------
# find_evidence_gaps
# ---------------------------------------------------------------------------


def test_peter_gallagher_has_approximate_birth_gap(db):
    gaps = find_evidence_gaps(db, PETER_GALLAGHER_ID)
    gap_types = {g.type for g in gaps}
    assert GapType.BIRTH_DATE_APPROXIMATE in gap_types


def test_peter_gallagher_has_unsourced_birth_gap(db):
    # Geni exports rarely carry source citations at the event level. This
    # asserts that the gap detector surfaces that reality.
    gaps = find_evidence_gaps(db, PETER_GALLAGHER_ID)
    gap_types = {g.type for g in gaps}
    assert GapType.BIRTH_DATE_UNSOURCED in gap_types


def test_gaps_per_individual_has_id(db):
    gaps = find_evidence_gaps(db, DAVID_ID)
    assert all(g.individual_id == DAVID_ID for g in gaps)
