"""Tests for ResearchState.

The core invariants: every transition returns a new instance, the prior
state is unchanged, and the history log grows by one entry per transition.
"""

from __future__ import annotations

from ancestors.agent.state import (
    ConfidenceLevel,
    DeadEnd,
    Fact,
    Hypothesis,
    new_state,
)


def test_new_state_is_empty():
    s = new_state("Did Peter Gallagher come from Donegal?")
    assert s.question.startswith("Did Peter")
    assert s.confirmed_facts == []
    assert s.working_hypotheses == []
    assert s.history == []


def test_add_fact_does_not_mutate_original():
    s0 = new_state("test")
    s1 = s0.add_fact(Fact(claim="x", confidence=ConfidenceLevel.PROBABLE))
    assert s0.confirmed_facts == []
    assert len(s1.confirmed_facts) == 1
    assert s1 is not s0


def test_history_grows_per_transition():
    s = new_state("test")
    s = s.add_fact(Fact(claim="a", confidence=ConfidenceLevel.PROBABLE))
    s = s.add_hypothesis(Hypothesis(claim="b", rationale="hunch"))
    s = s.add_open_question("c?")
    s = s.add_dead_end(DeadEnd(description="x", reason="nothing returned"))
    assert [t.kind for t in s.history] == [
        "add_fact",
        "add_hypothesis",
        "add_open_question",
        "add_dead_end",
    ]


def test_promote_hypothesis_moves_claim_to_facts():
    s = new_state("test").add_hypothesis(
        Hypothesis(claim="p", rationale="r", supporting=["src1"])
    )
    assert len(s.working_hypotheses) == 1
    s2 = s.promote_hypothesis(0, ConfidenceLevel.PROBABLE)
    assert s2.working_hypotheses == []
    assert len(s2.confirmed_facts) == 1
    fact = s2.confirmed_facts[0]
    assert fact.claim == "p"
    assert fact.confidence == ConfidenceLevel.PROBABLE
    assert fact.sources == ["src1"]


def test_close_open_question_removes_it():
    s = new_state("test")
    s = s.add_open_question("q1")
    s = s.add_open_question("q2")
    s = s.close_open_question(0)
    assert s.open_questions == ["q2"]


def test_summary_includes_question_and_fact_claim():
    s = new_state("Is X true?").add_fact(
        Fact(claim="a", confidence=ConfidenceLevel.CONFIRMED)
    )
    text = s.summary()
    assert "Is X true?" in text
    assert "a" in text  # the actual claim text must appear
    assert "confirmed" in text
