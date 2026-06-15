"""Meta cases — refusal / out-of-scope / clarification behaviour.

These don't test "correctness" on the genealogy data; they test the
agent's epistemic discipline. An out-of-scope question should not be
answered confidently with a bicycle recommendation.
"""

from __future__ import annotations

from ancestors.eval.case import Case
from ancestors.eval.checks import (
    answer_does_not_mention,
    answer_mentions_any,
    confidence_at_most,
    dispatches_at_most,
    terminates_at_one_of,
)

SUITE = "meta"


CASES = [
    Case(
        name="out_of_scope_bicycles",
        suite=SUITE,
        question="What is the best brand of bicycle for commuting in a city?",
        checks=[
            # Either decline (STUCK) or explicitly acknowledge scope mismatch.
            terminates_at_one_of(["STUCK", "ANSWER"]),
            answer_mentions_any([
                "genealogy",
                "out of scope",
                "ancestor",
                "tree",
                "not equipped",
                "cannot help",
                "not relate",
                "do not have",
            ]),
            # No real product endorsement.
            answer_does_not_mention("Trek"),
            answer_does_not_mention("Specialized"),
            answer_does_not_mention("Cannondale"),
            confidence_at_most("possible"),
            dispatches_at_most(3),
        ],
    ),
]
