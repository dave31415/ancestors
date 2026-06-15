"""Pattern cases — graph patterns the typed tools can't naturally express.

The cousin-marriage question is the canonical example: without SQL it
forced the agent to STUCK; with SQL it became answerable in a few turns.
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

SUITE = "pattern"


CASES = [
    Case(
        name="cousin_marriages",
        suite=SUITE,
        question=(
            "Did any of the people in this tree marry someone who is "
            "their first or second cousin?"
        ),
        checks=[
            terminates_at("ANSWER"),
            # The Crawford-Crawford match is the known second-cousin marriage.
            answer_mentions("Crawford"),
            answer_mentions_any(["second cousin", "2nd cousin"]),
            confidence_at_least("probable"),
            dispatches_at_most(10),
        ],
    ),
]
