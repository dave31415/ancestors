"""Detect evidence gaps in the GEDCOM database.

Evidence gaps are the deterministic feed for the agent's research agenda.
This module surfaces what is *known to be missing or weak* — the agent then
reasons about which gaps are worth closing given the research question.

Detection here is intentionally conservative. We flag absences, not judgments
about whether they should be filled. "Birth date missing" is a fact; whether
that fact matters depends on the question, and that is the agent's job.
"""

from __future__ import annotations

import logging

from ancestors.models import (
    EvidenceGap,
    GapType,
    GedcomDatabase,
    Individual,
)
from ancestors.tools.gedcom import IndividualNotFoundError

log = logging.getLogger(__name__)


def find_evidence_gaps(db: GedcomDatabase, id: str) -> list[EvidenceGap]:
    """Return the list of evidence gaps for a single individual."""
    if id not in db.individuals:
        raise IndividualNotFoundError(id)

    ind = db.individuals[id]
    gaps: list[EvidenceGap] = []

    _check_birth(ind, gaps)
    _check_death(ind, gaps)
    _check_parents(db, ind, gaps)
    _check_events_present(ind, gaps)

    return gaps


def _check_birth(ind: Individual, gaps: list[EvidenceGap]) -> None:
    if ind.birth is None or ind.birth.date is None:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.BIRTH_DATE_MISSING,
                description=f"{ind.display_name} has no recorded birth date.",
            )
        )
        return
    if ind.birth.date.is_approximate:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.BIRTH_DATE_APPROXIMATE,
                description=(
                    f"{ind.display_name} birth date is approximate "
                    f"('{ind.birth.date.raw}'); precise date undocumented."
                ),
            )
        )
    if not ind.birth.source_ids:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.BIRTH_DATE_UNSOURCED,
                description=(
                    f"{ind.display_name} birth date '{ind.birth.date.raw}' "
                    f"has no source citation."
                ),
            )
        )
    if ind.birth.place is None:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.BIRTH_PLACE_MISSING,
                description=f"{ind.display_name} has no recorded birth place.",
            )
        )


def _check_death(ind: Individual, gaps: list[EvidenceGap]) -> None:
    if ind.death is None or ind.death.date is None:
        # Don't flag living-or-unknown — if there's no death record AND we have
        # no reason to expect one, this isn't a gap. We use a 120-year heuristic:
        # if birth year is more than 120 years before the most-recent year we
        # see in the data, the absence of a death record IS a documented gap.
        # For now, only flag when there's no death event at all but a plausibly
        # deceased birth year. Simpler: flag missing death whenever birth year
        # is present and older than 120 years ago. We avoid hardcoding "now" by
        # using a conservative 1905 floor (anyone born before this is deceased).
        if ind.birth and ind.birth.date and ind.birth.date.year is not None:
            if ind.birth.date.year < 1905:
                gaps.append(
                    EvidenceGap(
                        individual_id=ind.id,
                        type=GapType.DEATH_DATE_MISSING,
                        description=(
                            f"{ind.display_name} born {ind.birth.date.year} "
                            f"has no death date — likely deceased but undocumented."
                        ),
                    )
                )
        return
    if not ind.death.source_ids:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.DEATH_DATE_UNSOURCED,
                description=(
                    f"{ind.display_name} death date '{ind.death.date.raw}' "
                    f"has no source citation."
                ),
            )
        )


def _check_parents(
    db: GedcomDatabase, ind: Individual, gaps: list[EvidenceGap]
) -> None:
    if not ind.families_as_child:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.PARENTS_UNKNOWN,
                description=f"{ind.display_name} has no parent family link.",
            )
        )
        return
    family = db.families.get(ind.families_as_child[0])
    if family is None:
        return
    if family.husband_id is None and family.wife_id is None:
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.PARENTS_UNKNOWN,
                description=(
                    f"{ind.display_name} parent family exists but neither "
                    f"parent is recorded."
                ),
            )
        )


def _check_events_present(ind: Individual, gaps: list[EvidenceGap]) -> None:
    if (
        ind.birth is None
        and ind.death is None
        and ind.burial is None
        and not ind.other_events
    ):
        gaps.append(
            EvidenceGap(
                individual_id=ind.id,
                type=GapType.NO_EVENTS,
                description=(
                    f"{ind.display_name} has no recorded life events at all."
                ),
            )
        )
