"""Lineage cases — graph walks and pairwise relationships.

Tests `get_ancestors_of`, `get_parents_of`, `find_common_ancestor`. The
agent should reach for typed tools here; SQL would be overkill.
"""

from __future__ import annotations

from ancestors.eval.case import Case
from ancestors.eval.checks import (
    answer_mentions,
    answer_mentions_all,
    answer_mentions_any,
    answer_mentions_year,
    confidence_at_least,
    dispatches_at_most,
    terminates_at,
)

SUITE = "lineage"


CASES = [
    Case(
        name="paternal_grandfather",
        suite=SUITE,
        question="Who is David Edmund Johnston's paternal grandfather, and where was he born?",
        checks=[
            terminates_at("ANSWER"),
            answer_mentions("John Dennis"),
            answer_mentions("Johnston"),
            answer_mentions_year(1885),
            answer_mentions("Newburg"),
            confidence_at_least("probable"),
            dispatches_at_most(15),
        ],
    ),
    Case(
        name="lancelot_johnston",
        suite=SUITE,
        question=(
            "How is David Edmund Johnston related to Dr. Lancelot Johnston, "
            "the Continental Congress surgeon born around 1748?"
        ),
        checks=[
            terminates_at("ANSWER"),
            # Lancelot is David's 4th-great-grandfather (6 generations apart).
            # Accept several common ways to phrase that.
            answer_mentions_any([
                "4th-great-grandfather",
                "great-great-great-great-grandfather",
                "5 greats",
                "fourth great-grandfather",
                "4th great-grandfather",
            ]),
            confidence_at_least("probable"),
            dispatches_at_most(12),
        ],
    ),
    Case(
        name="great_grandfathers",
        suite=SUITE,
        question="Name David Edmund Johnston's four great-grandfathers.",
        checks=[
            terminates_at("ANSWER"),
            # His four great-grandfathers, one per ancestral line.
            answer_mentions("Hugh Peter Johnston"),
            answer_mentions("Daniel Gallagher"),
            answer_mentions("Henry Joseph Keenan"),
            answer_mentions("John Wendelin Cronin"),
            confidence_at_least("probable"),
            dispatches_at_most(15),
        ],
    ),
]
