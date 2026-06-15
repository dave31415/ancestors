"""Lookup cases — find a person or simple fact.

These are the "easy" cases — typed tools should handle them in a small
number of dispatches. They're the floor: anything failing here suggests
something is wrong with the basic name-search pipeline.
"""

from __future__ import annotations

from ancestors.eval.case import Case
from ancestors.eval.checks import (
    answer_mentions,
    answer_mentions_all,
    answer_mentions_year,
    confidence_at_least,
    dispatches_at_most,
    terminates_at,
)

SUITE = "lookup"


CASES = [
    Case(
        name="david_johnston",
        suite=SUITE,
        question="Who is David Edmund Johnston?",
        checks=[
            terminates_at("ANSWER"),
            answer_mentions("David"),
            answer_mentions("Johnston"),
            answer_mentions_year(1975),
            confidence_at_least("probable"),
            dispatches_at_most(8),
        ],
    ),
    Case(
        name="peter_gallagher",
        suite=SUITE,
        question="Find Peter Gallagher, born around 1797. Tell me where he was born and when he died.",
        checks=[
            terminates_at("ANSWER"),
            answer_mentions("Peter"),
            answer_mentions("Gallagher"),
            answer_mentions("Tyrone"),
            answer_mentions_year(1883),
            confidence_at_least("probable"),
            dispatches_at_most(10),
        ],
    ),
    Case(
        name="earliest_born",
        suite=SUITE,
        question="Who is the person in the tree who was born the earliest? Give me their name and birth year.",
        checks=[
            terminates_at("ANSWER"),
            answer_mentions("Playters"),
            answer_mentions_year(1359),
            confidence_at_least("probable"),
            dispatches_at_most(6),
        ],
    ),
]
