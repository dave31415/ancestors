"""Genealogy tool registry — argument schemas + the agent-facing whitelist.

This file defines *what* the agent can call. The Dispatcher in dispatch.py
defines *how* the call is validated, memoised, and executed. The two are
deliberately separable: the dispatcher is generic and would work for any
domain; this file is the genealogy side that names the tools, declares
their typed argument schemas, and binds each name to a `dsl` function.

Adding a new tool is a one-stop edit here:

  1. Define its argument schema (a Pydantic model below).
  2. Add a `Tool(...)` entry to TOOLS.
  3. Implement the function in dsl.py.

The argument schemas are not idle annotations — they are the only
validation between the model's free-form output and the tool function.
Every constraint here (length bound, regex, enum) is load-bearing for
defence-in-depth.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from ancestors import dsl
from ancestors.dispatch import (
    MAX_HYDRATION_LIMIT,
    MAX_STRING_ARG_LEN,
    Tool,
)
from ancestors.models import EventType, Sex, SortKey

# Person ids are GEDCOM-shaped: @XXXXXX@ with uppercase alphanumerics.
PERSON_ID_RE = re.compile(r"^@[A-Z][A-Z0-9_\-]{0,200}@$")

PersonId = Annotated[str, StringConstraints(pattern=PERSON_ID_RE.pattern, max_length=204)]
SetHandle = Annotated[str, StringConstraints(pattern=r"^h_[0-9]+$", max_length=16)]
BoundedStr = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_STRING_ARG_LEN)
]
Generations = Annotated[int, Field(ge=0, le=30)]
HydrationLimit = Annotated[int, Field(ge=1, le=MAX_HYDRATION_LIMIT)]


# ---------------------------------------------------------------------------
# Per-tool argument schemas. Every tool has one, even if it takes no args.
# These ARE the agent-facing input contract.
# ---------------------------------------------------------------------------


class _NoArgs(BaseModel):
    model_config = {"extra": "forbid"}


class _PersonOnly(BaseModel):
    model_config = {"extra": "forbid"}
    person_id: PersonId


class _AncestorsArgs(BaseModel):
    model_config = {"extra": "forbid"}
    person_id: PersonId
    max_generations: Generations = 20


class _CommonAncestorArgs(BaseModel):
    model_config = {"extra": "forbid"}
    a_id: PersonId
    b_id: PersonId
    max_generations: Generations = 20


class _SetOnly(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle


class _TwoSets(BaseModel):
    model_config = {"extra": "forbid"}
    a: SetHandle
    b: SetHandle


class _FilterSurname(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    surname: BoundedStr


class _FilterGiven(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    fragment: BoundedStr


class _FilterSex(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    sex: Sex


class _FilterEventPlace(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    event_type: EventType
    place_contains: BoundedStr


class _FilterEventYearRange(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    event_type: EventType
    year_min: int | None = Field(default=None, ge=1, le=9999)
    year_max: int | None = Field(default=None, ge=1, le=9999)


class _FilterEventTypeOnly(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    event_type: EventType


class _Hydrate(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    limit: HydrationLimit = 25


class _GroupBy(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    by: Annotated[str, StringConstraints(pattern=r"^(generation|birth_century|birth_country|sex)$")]


class _SortBy(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    by: SortKey
    descending: bool = False
    limit: Annotated[int, Field(ge=1, le=MAX_HYDRATION_LIMIT)] | None = None


# Length-bounded so the agent can't smuggle a megabyte query past validation.
# 5K chars comfortably handles recursive CTEs the agent would plausibly write.
MAX_SQL_LEN = 5000


class _RunSql(BaseModel):
    model_config = {"extra": "forbid"}
    query: Annotated[
        str,
        StringConstraints(min_length=1, max_length=MAX_SQL_LEN),
    ]


# ---------------------------------------------------------------------------
# Registry — the whitelist. Only tools listed here are dispatchable.
# ---------------------------------------------------------------------------


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("all_individuals", _NoArgs, dsl.all_individuals,
             "Return an IdSet of every individual in the database."),
        Tool("get_ancestors_of", _AncestorsArgs, dsl.get_ancestors_of,
             "Ancestors of person_id, root included, with generation metadata."),
        Tool("get_descendants_of", _AncestorsArgs, dsl.get_descendants_of,
             "Descendants of person_id, root included, with generation metadata."),
        Tool("get_parents_of", _PersonOnly, dsl.get_parents_of,
             "Direct parents (husband+wife of person's child-family)."),
        Tool("get_children_of", _PersonOnly, dsl.get_children_of,
             "Direct children across all spouse-families."),
        Tool("get_siblings_of", _PersonOnly, dsl.get_siblings_of,
             "Direct siblings (children of the same parent-family)."),
        Tool("get_spouses_of", _PersonOnly, dsl.get_spouses_of,
             "All recorded spouses across spouse-families."),
        Tool("find_common_ancestor", _CommonAncestorArgs, dsl.find_common_ancestor,
             "Find the most recent common ancestor of two named individuals. Returns ancestor id+name, generation distance on each side, and a kinship label (siblings, first cousins, etc.). Returns nulls when no common ancestor exists within max_generations."),
        Tool("intersect", _TwoSets, dsl.intersect,
             "Set intersection. Metadata from a wins on overlap."),
        Tool("union", _TwoSets, dsl.union,
             "Set union, preserving first-seen order."),
        Tool("difference", _TwoSets, dsl.difference,
             "Elements of a not in b."),
        Tool("filter_by_surname", _FilterSurname, dsl.filter_by_surname,
             "Keep only ids whose primary name surname matches exactly."),
        Tool("filter_by_given_name_contains", _FilterGiven, dsl.filter_by_given_name_contains,
             "Keep only ids whose given name contains the fragment."),
        Tool("filter_by_sex", _FilterSex, dsl.filter_by_sex,
             "Keep only ids matching the specified sex."),
        Tool("filter_by_event_place", _FilterEventPlace, dsl.filter_by_event_place,
             "Keep ids with at least one event of type T whose place mentions the string."),
        Tool("filter_by_event_year_range", _FilterEventYearRange, dsl.filter_by_event_year_range,
             "Keep ids with at least one event of type T in the given year range."),
        Tool("filter_has_no_source", _FilterEventTypeOnly, dsl.filter_has_no_source,
             "Keep ids whose event of type T exists but has no source citation."),
        Tool("sort_by", _SortBy, dsl.sort_by,
             "Order an IdSet by birth_year/death_year/lifespan/surname/given_name, optionally taking the top N. Missing values drop from the result."),
        Tool("count", _SetOnly, dsl.count,
             "Return the size of an IdSet."),
        Tool("group_by", _GroupBy, dsl.group_by,
             "Aggregate counts by generation/birth_century/birth_country/sex."),
        Tool("get_individuals", _Hydrate, dsl.get_individuals,
             "Hydrate up to `limit` ids into full Individual records."),
        Tool("get_summary", _PersonOnly, dsl.get_summary,
             "Compact, deterministic one-line factual précis of a person."),
        Tool("get_evidence_gaps", _PersonOnly, dsl.get_evidence_gaps,
             "List evidence-gap types for a person (driving the research agenda)."),
        Tool("run_sql", _RunSql, dsl.run_sql,
             "Execute one read-only SQL statement against the corpus. Use for "
             "aggregations, graph patterns (recursive CTEs), or queries the "
             "typed tools can't express. The schema is included in the system "
             "prompt. Prefer typed tools for narrow questions (named lookups, "
             "single-pair MRCA, sorting) — they are cheaper and more auditable."),
    ]
}


def list_tools() -> list[dict[str, str]]:
    """Schema-level introspection — what the agent would be told it can do."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "args_schema": str(t.args_model.model_json_schema()),
        }
        for t in TOOLS.values()
    ]
