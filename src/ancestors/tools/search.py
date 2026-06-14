"""Search and ranking tools over the in-memory GEDCOM database.

Scoring lives here, not in the agent. The agent reasons about scores; it does
not compute them. Per-field breakdowns are returned alongside the overall
score so the agent can weigh evidence transparently.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from ancestors.models import (
    GedcomDatabase,
    Individual,
    IndividualMatch,
    IndividualQuery,
    MatchScore,
    Sex,
)

log = logging.getLogger(__name__)

# Per-field weights used to combine sub-scores into an overall match score.
# Tuned to reflect what genealogists weight most heavily in a candidate match:
# surname/given name first, then date plausibility, then place, then sex.
_WEIGHTS = {"name": 0.50, "year": 0.25, "place": 0.15, "sex": 0.10}

# Threshold below which a match is not worth surfacing at all. Set so that
# single-field name queries don't dredge up unrelated surnames that happen to
# share a few characters under SequenceMatcher. With this threshold an exact
# surname match (1.0) and very close fuzzy matches survive; loose lexical
# coincidences are dropped.
_MIN_SCORE = 0.6


def find_individuals(
    db: GedcomDatabase, query: IndividualQuery
) -> list[IndividualMatch]:
    """Return ranked candidate matches for the query.

    Empty queries return [] — there is nothing to score against. Candidates
    scoring below _MIN_SCORE are dropped. Results are sorted by overall score
    desc, then by id for stability.
    """
    if _query_is_empty(query):
        return []

    matches: list[IndividualMatch] = []
    for ind in db.individuals.values():
        score = _score_individual(ind, query)
        if score.overall >= _MIN_SCORE:
            matches.append(IndividualMatch(individual=ind, score=score))

    matches.sort(key=lambda m: (-m.score.overall, m.individual.id))
    return matches[: query.limit]


def _query_is_empty(q: IndividualQuery) -> bool:
    return not any(
        [
            q.given,
            q.surname,
            q.sex,
            q.birth_year_min,
            q.birth_year_max,
            q.death_year_min,
            q.death_year_max,
            q.place_contains,
        ]
    )


def _score_individual(ind: Individual, q: IndividualQuery) -> MatchScore:
    name = _score_name(ind, q)
    year = _score_year(ind, q)
    place = _score_place(ind, q)
    sex = _score_sex(ind, q)

    used: list[tuple[str, float]] = []
    if name is not None:
        used.append(("name", name))
    if year is not None:
        used.append(("year", year))
    if place is not None:
        used.append(("place", place))
    if sex is not None:
        used.append(("sex", sex))

    if not used:
        return MatchScore(overall=0.0, name=name, year=year, place=place, sex=sex)

    total_weight = sum(_WEIGHTS[k] for k, _ in used)
    overall = sum(_WEIGHTS[k] * v for k, v in used) / total_weight

    return MatchScore(overall=overall, name=name, year=year, place=place, sex=sex)


def _score_name(ind: Individual, q: IndividualQuery) -> float | None:
    if not q.given and not q.surname:
        return None

    primary = ind.primary_name
    if primary is None:
        return 0.0

    parts: list[float] = []
    if q.surname:
        ind_surname = (primary.surname or "").lower()
        parts.append(_string_similarity(q.surname.lower(), ind_surname))
    if q.given:
        ind_given = (primary.given or "").lower()
        parts.append(_string_similarity(q.given.lower(), ind_given))

    return sum(parts) / len(parts) if parts else None


def _score_year(ind: Individual, q: IndividualQuery) -> float | None:
    constraints: list[float] = []

    if q.birth_year_min is not None or q.birth_year_max is not None:
        constraints.append(
            _year_range_score(
                _event_year(ind, "birth"), q.birth_year_min, q.birth_year_max
            )
        )
    if q.death_year_min is not None or q.death_year_max is not None:
        constraints.append(
            _year_range_score(
                _event_year(ind, "death"), q.death_year_min, q.death_year_max
            )
        )

    return sum(constraints) / len(constraints) if constraints else None


def _event_year(ind: Individual, which: str) -> int | None:
    event = ind.birth if which == "birth" else ind.death
    if event is None or event.date is None:
        return None
    return event.date.year


def _year_range_score(year: int | None, lo: int | None, hi: int | None) -> float:
    if year is None:
        return 0.0
    if lo is not None and hi is not None:
        if lo <= year <= hi:
            return 1.0
        # Linear decay: 1 year off = 0.9, 10 years off = 0.0 (clamped).
        distance = lo - year if year < lo else year - hi
        return max(0.0, 1.0 - distance / 10.0)
    if lo is not None:
        return 1.0 if year >= lo else max(0.0, 1.0 - (lo - year) / 10.0)
    if hi is not None:
        return 1.0 if year <= hi else max(0.0, 1.0 - (year - hi) / 10.0)
    return 0.0


def _score_place(ind: Individual, q: IndividualQuery) -> float | None:
    if not q.place_contains:
        return None
    needle = q.place_contains.lower()
    haystacks: list[str] = []
    for event in (ind.birth, ind.death, ind.burial, *ind.other_events):
        if event is None or event.place is None:
            continue
        p = event.place
        for field in (p.raw, p.city, p.state, p.country):
            if field:
                haystacks.append(field.lower())
    if not haystacks:
        return 0.0
    return 1.0 if any(needle in h for h in haystacks) else 0.0


def _score_sex(ind: Individual, q: IndividualQuery) -> float | None:
    if q.sex is None:
        return None
    if ind.sex == Sex.UNKNOWN:
        # Can't penalize an unknown sex against a hard constraint without false
        # negatives — treat as neutral evidence.
        return 0.5
    return 1.0 if ind.sex == q.sex else 0.0


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()
