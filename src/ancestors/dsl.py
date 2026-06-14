"""The agent-facing DSL — typed, composable, no Python execution required.

Every callable here takes JSON-friendly arguments (strings, ints, enums, and
IdSets) and returns a JSON-friendly result. An LLM agent composes these by
choosing which to call and feeding the output of one into the input of
another. It does not write code. It does not assemble queries.

Three families:
    Producers   — generate an IdSet from the graph or record store.
    Refiners    — filter or combine IdSets. Output is always smaller-or-equal.
    Readers     — non-mutating inspection of an IdSet.

Tools delegate to the existing ancestors.tools.* implementations. The DSL is
the agent surface; the tools layer is the engine. This keeps the algebraic
shape separate from the storage / traversal details, and lets either evolve
without disturbing the other.
"""

from __future__ import annotations

from collections import Counter

from ancestors.models import (
    AncestorNode,
    DescendantNode,
    EventType,
    GapType,
    IdSet,
    Individual,
    IndividualQuery,
    Sex,
)
from ancestors.session import current_session
from ancestors.tools.gaps import find_evidence_gaps
from ancestors.tools.lineage import get_ancestors, get_descendants
from ancestors.tools.search import find_individuals

# ---------------------------------------------------------------------------
# Producers — build an IdSet from the graph or record store.
# ---------------------------------------------------------------------------


def all_individuals() -> IdSet:
    """The set of every known individual.

    The starting point for unrestricted searches. Combine with refiners to
    narrow before reading.
    """
    db = current_session().db
    return IdSet(ids=list(db.individuals.keys()))


def get_ancestors_of(person_id: str, max_generations: int = 20) -> IdSet:
    """Ancestors of `person_id`, walked up to `max_generations`.

    The root (person_id itself) is included. Each id carries `generation`
    metadata (0 = root, 1 = parents, 2 = grandparents, ...).
    """
    db = current_session().db
    tree = get_ancestors(db, person_id, max_generations)
    ids: list[str] = []
    metadata: dict[str, dict[str, object]] = {}

    def walk(node: AncestorNode) -> None:
        ids.append(node.individual.id)
        metadata[node.individual.id] = {"generation": node.generation}
        if node.father is not None:
            walk(node.father)
        if node.mother is not None:
            walk(node.mother)

    walk(tree)
    return IdSet(ids=ids, metadata=metadata)


def get_descendants_of(person_id: str, max_generations: int = 20) -> IdSet:
    """Descendants of `person_id`, walked up to `max_generations`.

    Root included; each id carries `generation` (0 = root, 1 = children, ...).
    """
    db = current_session().db
    tree = get_descendants(db, person_id, max_generations)
    ids: list[str] = []
    metadata: dict[str, dict[str, object]] = {}

    def walk(node: DescendantNode) -> None:
        ids.append(node.individual.id)
        metadata[node.individual.id] = {"generation": node.generation}
        for child in node.children:
            walk(child)

    walk(tree)
    return IdSet(ids=ids, metadata=metadata)


def get_parents_of(person_id: str) -> IdSet:
    db = current_session().db
    ind = db.individuals[person_id]
    if not ind.families_as_child:
        return IdSet()
    fam = db.families.get(ind.families_as_child[0])
    if fam is None:
        return IdSet()
    ids = [pid for pid in (fam.husband_id, fam.wife_id) if pid]
    return IdSet(ids=ids)


def get_children_of(person_id: str) -> IdSet:
    db = current_session().db
    ind = db.individuals[person_id]
    seen: set[str] = set()
    ids: list[str] = []
    for fam_id in ind.families_as_spouse:
        fam = db.families.get(fam_id)
        if fam is None:
            continue
        for cid in fam.children_ids:
            if cid not in seen and cid in db.individuals:
                seen.add(cid)
                ids.append(cid)
    return IdSet(ids=ids)


def get_siblings_of(person_id: str) -> IdSet:
    db = current_session().db
    ind = db.individuals[person_id]
    seen: set[str] = set()
    ids: list[str] = []
    for fam_id in ind.families_as_child:
        fam = db.families.get(fam_id)
        if fam is None:
            continue
        for cid in fam.children_ids:
            if cid != person_id and cid not in seen and cid in db.individuals:
                seen.add(cid)
                ids.append(cid)
    return IdSet(ids=ids)


def get_spouses_of(person_id: str) -> IdSet:
    db = current_session().db
    ind = db.individuals[person_id]
    seen: set[str] = set()
    ids: list[str] = []
    for fam_id in ind.families_as_spouse:
        fam = db.families.get(fam_id)
        if fam is None:
            continue
        for sid in (fam.husband_id, fam.wife_id):
            if sid and sid != person_id and sid not in seen and sid in db.individuals:
                seen.add(sid)
                ids.append(sid)
    return IdSet(ids=ids)


# ---------------------------------------------------------------------------
# Set operations — combine two IdSets. Metadata from `a` wins on overlap.
# ---------------------------------------------------------------------------


def intersect(a: IdSet, b: IdSet) -> IdSet:
    bset = set(b.ids)
    ids = [i for i in a.ids if i in bset]
    return IdSet(ids=ids, metadata={i: a.metadata.get(i, {}) for i in ids})


def union(a: IdSet, b: IdSet) -> IdSet:
    seen: set[str] = set()
    ids: list[str] = []
    meta: dict[str, dict[str, object]] = {}
    for i in (*a.ids, *b.ids):
        if i in seen:
            continue
        seen.add(i)
        ids.append(i)
        meta[i] = a.metadata.get(i) or b.metadata.get(i, {})
    return IdSet(ids=ids, metadata=meta)


def difference(a: IdSet, b: IdSet) -> IdSet:
    bset = set(b.ids)
    ids = [i for i in a.ids if i not in bset]
    return IdSet(ids=ids, metadata={i: a.metadata.get(i, {}) for i in ids})


# ---------------------------------------------------------------------------
# Refiners — narrow an IdSet by a typed predicate against record-store data.
# ---------------------------------------------------------------------------


def _hydrate(ids: IdSet) -> list[Individual]:
    db = current_session().db
    return [db.individuals[i] for i in ids.ids if i in db.individuals]


def _restrict(ids: IdSet, keep: list[str]) -> IdSet:
    keep_set = set(keep)
    kept_ids = [i for i in ids.ids if i in keep_set]
    return IdSet(
        ids=kept_ids,
        metadata={i: ids.metadata.get(i, {}) for i in kept_ids},
    )


def filter_by_surname(ids: IdSet, surname: str) -> IdSet:
    target = surname.strip().lower()
    keep = [
        ind.id
        for ind in _hydrate(ids)
        if ind.primary_name and (ind.primary_name.surname or "").lower() == target
    ]
    return _restrict(ids, keep)


def filter_by_given_name_contains(ids: IdSet, fragment: str) -> IdSet:
    needle = fragment.strip().lower()
    keep = [
        ind.id
        for ind in _hydrate(ids)
        if ind.primary_name and needle in (ind.primary_name.given or "").lower()
    ]
    return _restrict(ids, keep)


def filter_by_sex(ids: IdSet, sex: Sex) -> IdSet:
    keep = [ind.id for ind in _hydrate(ids) if ind.sex == sex]
    return _restrict(ids, keep)


def filter_by_event_place(
    ids: IdSet, event_type: EventType, place_contains: str
) -> IdSet:
    needle = place_contains.strip().lower()
    keep: list[str] = []
    for ind in _hydrate(ids):
        for ev in _events_of_type(ind, event_type):
            if ev.place is None:
                continue
            for field in (ev.place.country, ev.place.state, ev.place.city, ev.place.raw):
                if field and needle in field.lower():
                    keep.append(ind.id)
                    break
            else:
                continue
            break
    return _restrict(ids, keep)


def filter_by_event_year_range(
    ids: IdSet,
    event_type: EventType,
    year_min: int | None = None,
    year_max: int | None = None,
) -> IdSet:
    keep: list[str] = []
    for ind in _hydrate(ids):
        for ev in _events_of_type(ind, event_type):
            if ev.date is None or ev.date.year is None:
                continue
            y = ev.date.year
            if year_min is not None and y < year_min:
                continue
            if year_max is not None and y > year_max:
                continue
            keep.append(ind.id)
            break
    return _restrict(ids, keep)


def filter_has_no_source(ids: IdSet, event_type: EventType) -> IdSet:
    """Individuals whose specified event exists but carries no source citation.

    Surfaces the agent's research agenda: "born before 1800, no source on
    birth" is exactly the population a researcher needs to prioritize.
    """
    keep: list[str] = []
    for ind in _hydrate(ids):
        events = list(_events_of_type(ind, event_type))
        if not events:
            continue
        if all(not ev.source_ids for ev in events):
            keep.append(ind.id)
    return _restrict(ids, keep)


def filter_by_query(ids: IdSet, query: IndividualQuery) -> IdSet:
    """Apply the full IndividualQuery (name+date+place+sex) via the scorer.

    Convenience for queries that combine many constraints at once. Uses the
    same scoring threshold as find_individuals.
    """
    db = current_session().db
    matches = find_individuals(db, query)
    matched_ids = {m.individual.id for m in matches}
    keep = [i for i in ids.ids if i in matched_ids]
    return _restrict(ids, keep)


def _events_of_type(ind: Individual, event_type: EventType):
    if event_type == EventType.BIRTH and ind.birth is not None:
        yield ind.birth
    elif event_type == EventType.DEATH and ind.death is not None:
        yield ind.death
    elif event_type == EventType.BURIAL and ind.burial is not None:
        yield ind.burial
    else:
        for ev in ind.other_events:
            if ev.type == event_type:
                yield ev


# ---------------------------------------------------------------------------
# Readers — inspect an IdSet without changing it.
# ---------------------------------------------------------------------------


def count(ids: IdSet) -> int:
    return ids.size


def group_by(ids: IdSet, by: str) -> dict[str, int]:
    """Aggregate counts. `by` selects the grouping function:

    - "generation": uses producer-attached metadata (ancestors/descendants).
    - "birth_century": derived from birth.date.year.
    - "birth_country": derived from birth.place.country.
    - "sex": from individual.sex.
    """
    if by == "generation":
        counter: Counter[str] = Counter()
        for i in ids.ids:
            g = ids.with_meta(i, "generation")
            counter[str(g) if g is not None else "unknown"] += 1
        return dict(counter)

    individuals = _hydrate(ids)
    if by == "birth_century":
        counter = Counter()
        for ind in individuals:
            y = ind.birth.date.year if ind.birth and ind.birth.date else None
            counter[f"{y // 100 + 1}c" if y else "unknown"] += 1
        return dict(counter)
    if by == "birth_country":
        counter = Counter()
        for ind in individuals:
            c = (
                ind.birth.place.country
                if ind.birth and ind.birth.place and ind.birth.place.country
                else None
            )
            counter[c or "unknown"] += 1
        return dict(counter)
    if by == "sex":
        return dict(Counter(ind.sex.value for ind in individuals))

    raise ValueError(f"unknown group_by key: {by!r}")


def get_individuals(ids: IdSet, limit: int = 25) -> list[Individual]:
    """Hydrate up to `limit` ids into full Individual records.

    The dispatcher caps `limit` at the registered upper bound; an agent
    asking for "all 10,000 records" will be silently truncated to the
    registered ceiling.
    """
    db = current_session().db
    out: list[Individual] = []
    for i in ids.ids[:limit]:
        ind = db.individuals.get(i)
        if ind is not None:
            out.append(ind)
    return out


def get_summary(person_id: str) -> str:
    """Compact, hand-written précis suitable for an LLM prompt.

    Deliberately deterministic — no LLM in the formatting loop, no place for
    confabulation to creep into a "background" string the agent then reasons
    against.
    """
    db = current_session().db
    ind = db.individuals[person_id]
    parts: list[str] = [ind.display_name]
    if ind.birth and ind.birth.date:
        parts.append(f"b. {ind.birth.date.raw}")
        if ind.birth.place:
            loc = _format_place(ind.birth.place)
            if loc:
                parts.append(f"in {loc}")
    if ind.death and ind.death.date:
        parts.append(f"d. {ind.death.date.raw}")
        if ind.death.place:
            loc = _format_place(ind.death.place)
            if loc:
                parts.append(f"in {loc}")
    return " ".join(parts)


def get_evidence_gaps(person_id: str) -> list[GapType]:
    db = current_session().db
    return [g.type for g in find_evidence_gaps(db, person_id)]


def _format_place(place) -> str:
    parts = [p for p in (place.city, place.state, place.country) if p]
    if parts:
        return ", ".join(parts)
    return place.raw or ""
