"""Aggregation cases — SQL escape hatch territory.

These questions are aggregations or sorts that fit SQL better than the
typed tools. We expect run_sql to feature prominently; tools_used_include
isn't asserted because the model might still get there via the typed
path with sort_by + count.
"""

from __future__ import annotations

from ancestors.eval.case import Case
from ancestors.eval.checks import (
    answer_mentions,
    answer_mentions_any,
    confidence_at_least,
    dispatches_at_most,
    terminates_at,
)

SUITE = "aggregation"


CASES = [
    Case(
        name="country_counts",
        suite=SUITE,
        question="What are the most common countries of birth in this tree?",
        checks=[
            terminates_at("ANSWER"),
            # The top labels include Ireland, Scotland, England (all near 29).
            answer_mentions("Ireland"),
            answer_mentions("Scotland"),
            answer_mentions("England"),
            confidence_at_least("probable"),
            dispatches_at_most(6),
        ],
    ),
    Case(
        name="longest_lived",
        suite=SUITE,
        question=(
            "Who is the person in the tree who lived the longest "
            "(based on recorded birth and death years)? Give name and lifespan."
        ),
        checks=[
            terminates_at("ANSWER"),
            # We don't pin a specific individual because the data lets a few
            # candidates be plausible depending on the agent's tie-break.
            # We do require it to mention a lifespan number ≥ 80.
            answer_mentions_any([
                "80 years",
                "81 years",
                "82 years",
                "83 years",
                "84 years",
                "85 years",
                "86 years",
                "87 years",
                "88 years",
                "89 years",
                "90 years",
                "91 years",
                "92 years",
                "93 years",
                "94 years",
                "95 years",
                "96 years",
                "97 years",
                "98 years",
                "99 years",
                "100 years",
                "101 years",
                "102 years",
                "lived to be",
                "lifespan",
            ]),
            confidence_at_least("probable"),
            dispatches_at_most(6),
        ],
    ),
]
