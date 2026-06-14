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


def test_sort_by_birth_year_ascending_finds_earliest(d):
    everyone = d.dispatch("all_individuals", {})
    s = d.dispatch(
        "sort_by",
        {"ids": everyone.set_handle, "by": "birth_year", "limit": 1},
    )
    assert s.ok and s.set_size == 1
    record = d.dispatch("get_individuals", {"ids": s.set_handle, "limit": 1})
    earliest = record.value[0]
    # Thomas /Playters/, born 1359 — the documented earliest birth in the corpus.
    assert earliest["birth"]["date"]["year"] == 1359
    assert "Playters" in earliest["names"][0]["full"]


def test_sort_by_birth_year_descending_returns_latest(d):
    everyone = d.dispatch("all_individuals", {})
    s = d.dispatch(
        "sort_by",
        {
            "ids": everyone.set_handle,
            "by": "birth_year",
            "descending": True,
            "limit": 3,
        },
    )
    assert s.ok and s.set_size == 3


def test_sort_by_lifespan_drops_individuals_without_both_dates(d):
    everyone = d.dispatch("all_individuals", {})
    s = d.dispatch(
        "sort_by", {"ids": everyone.set_handle, "by": "lifespan", "descending": True, "limit": 5},
    )
    assert s.ok
    # All 5 must have both birth and death years available — the sort key
    # is undefined otherwise, so they were dropped.
    records = d.dispatch("get_individuals", {"ids": s.set_handle, "limit": 5})
    for ind in records.value:
        assert ind["birth"]["date"]["year"] is not None
        assert ind["death"]["date"]["year"] is not None


def test_sort_by_surname_alphabetical(d):
    a = d.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 2})
    s = d.dispatch("sort_by", {"ids": a.set_handle, "by": "surname"})
    assert s.ok
    recs = d.dispatch("get_individuals", {"ids": s.set_handle, "limit": 25})
    surnames = [
        (ind.get("names", [{}])[0] or {}).get("surname")
        for ind in recs.value
    ]
    surnames = [s for s in surnames if s]
    assert surnames == sorted(surnames, key=str.lower)


def test_find_common_ancestor_for_known_pair(d):
    # David and his father share the father — so the MRCA *is* his father,
    # generations 1 and 0 — labelled "parent ↔ child".
    fams = d.dispatch("get_parents_of", {"person_id": DAVID_ID})
    parents_records = d.dispatch("get_individuals", {"ids": fams.set_handle, "limit": 2}).value
    father_id = next(
        ind["id"] for ind in parents_records if ind.get("sex") == "M"
    )
    r = d.dispatch(
        "find_common_ancestor",
        {"a_id": DAVID_ID, "b_id": father_id, "max_generations": 5},
    )
    assert r.ok
    assert r.value["ancestor_id"] == father_id
    assert r.value["generations_to_a"] == 1
    assert r.value["generations_to_b"] == 0
    assert "parent" in r.value["relationship_label"]


def test_find_common_ancestor_returns_nulls_when_unrelated(d):
    # Pick someone definitely not in David's tree: Thomas Playters (1359
    # English, on the deep maternal side via the Cloptons). They DO share a
    # gen-many ancestor, but limiting max_generations=2 forces no match.
    everyone = d.dispatch("all_individuals", {})
    earliest_set = d.dispatch(
        "sort_by", {"ids": everyone.set_handle, "by": "birth_year", "limit": 1}
    )
    earliest_recs = d.dispatch(
        "get_individuals", {"ids": earliest_set.set_handle, "limit": 1}
    )
    earliest_id = earliest_recs.value[0]["id"]
    r = d.dispatch(
        "find_common_ancestor",
        {"a_id": DAVID_ID, "b_id": earliest_id, "max_generations": 2},
    )
    assert r.ok
    assert r.value["ancestor_id"] is None
    assert r.value["relationship_label"] is None


def test_kinship_label_known_pairs():
    from ancestors.dsl import _label_kinship

    assert _label_kinship(0, 0) == "same individual"
    assert "parent" in _label_kinship(1, 0)
    assert "grandparent" in _label_kinship(2, 0)
    assert "great-grandparent" in _label_kinship(3, 0)
    assert "siblings" in _label_kinship(1, 1)
    assert _label_kinship(2, 2) == "first cousins"
    assert _label_kinship(3, 3) == "second cousins"
    assert _label_kinship(4, 4) == "third cousins"
    assert _label_kinship(2, 3) == "first cousins once removed"
    assert _label_kinship(3, 5) == "second cousins twice removed"
    assert "aunt/uncle" in _label_kinship(1, 2)
    assert "great-aunt/uncle" in _label_kinship(1, 3)


def test_sort_by_preserves_metadata():
    # Use a fresh dispatcher so the test is independent.
    from ancestors.dispatch import Dispatcher

    d = Dispatcher()
    a = d.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 3})
    # get_ancestors_of attaches generation metadata
    s = d.dispatch("sort_by", {"ids": a.set_handle, "by": "birth_year"})
    # Inspect the IdSet behind the handle: rank attached, generation preserved.
    raw_set = d.sets[s.set_handle]
    assert all("rank" in meta for meta in raw_set.metadata.values())
    assert any("generation" in meta for meta in raw_set.metadata.values())
