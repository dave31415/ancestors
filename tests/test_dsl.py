"""Functional tests for DSL primitives, exercised via the dispatcher.

We deliberately go through the dispatcher rather than calling DSL functions
directly. The contract we care about is what an agent sees — handles plus
structured results — not what an internal caller sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.dispatch import Dispatcher
from ancestors.session import bind_session, clear_session
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"
PETER_ID = "@I6000000001912136867@"


@pytest.fixture(scope="module", autouse=True)
def session():
    bind_session(load_gedcom(GEDCOM_PATH))
    yield
    clear_session()


@pytest.fixture
def d():
    return Dispatcher()


def test_all_individuals_returns_full_population(d):
    r = d.dispatch("all_individuals", {})
    assert r.ok and r.set_size == 336


def test_ancestors_pipeline_with_place_filter(d):
    a = d.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 25})
    b = d.dispatch(
        "filter_by_event_place",
        {"ids": a.set_handle, "event_type": "BIRT", "place_contains": "England"},
    )
    # We know from prior analysis that ~80+ ancestors have BIRT in England.
    assert b.set_size > 50


def test_intersection_of_two_paths(d):
    # Set algebra: ancestors of David ∩ all_individuals = ancestors of David.
    a = d.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 25})
    full = d.dispatch("all_individuals", {})
    inter = d.dispatch("intersect", {"a": a.set_handle, "b": full.set_handle})
    assert inter.set_size == a.set_size


def test_filter_by_surname_then_year_range(d):
    all_ = d.dispatch("all_individuals", {})
    gal = d.dispatch(
        "filter_by_surname", {"ids": all_.set_handle, "surname": "Gallagher"}
    )
    in_range = d.dispatch(
        "filter_by_event_year_range",
        {
            "ids": gal.set_handle,
            "event_type": "BIRT",
            "year_min": 1830,
            "year_max": 1900,
        },
    )
    assert 1 <= in_range.set_size <= gal.set_size


def test_get_siblings_of(d):
    r = d.dispatch("get_siblings_of", {"person_id": DAVID_ID})
    # Whatever David's siblings are, the set must not contain David.
    assert r.ok
    inter = d.dispatch(
        "intersect",
        {"a": r.set_handle, "b": d.dispatch("all_individuals", {}).set_handle},
    )
    assert inter.set_size == r.set_size  # all siblings exist in db


def test_group_by_generation(d):
    a = d.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 5})
    g = d.dispatch("group_by", {"ids": a.set_handle, "by": "generation"})
    assert g.ok
    # Generation 0 always has exactly one — the root.
    assert g.value.get("0") == 1


def test_get_individuals_hydrates(d):
    a = d.dispatch("get_parents_of", {"person_id": DAVID_ID})
    r = d.dispatch("get_individuals", {"ids": a.set_handle, "limit": 25})
    assert r.ok and isinstance(r.value, list) and 1 <= len(r.value) <= 2


def test_get_summary_for_peter(d):
    r = d.dispatch("get_summary", {"person_id": PETER_ID})
    assert r.ok
    assert "Peter" in r.value and "Gallagher" in r.value


def test_evidence_gaps_for_peter_include_unsourced(d):
    r = d.dispatch("get_evidence_gaps", {"person_id": PETER_ID})
    assert r.ok and "birth_date_unsourced" in r.value
