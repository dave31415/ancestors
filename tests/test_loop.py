"""Tests for the agent loop's state machine.

Uses scripted Planner and Assessor stubs in place of LLM-backed ones, so
the tests are fast, deterministic, and don't touch the API. The real LLM
integrations land in subsequent steps and will be tested separately.

What we verify here:

- Linear path through PLAN -> EXECUTE -> ASSESS -> ANSWER works.
- Multi-call plans execute every call before assessing.
- Revising a plan returns control to PLAN.
- Assessor-declared stuck terminates with STUCK.
- Hard dispatch cap forces STUCK.
- Stall detector catches a repeating tool pattern.
- Tool dispatch failures are surfaced to ASSESS in recent_results.
- Facts/hypotheses/dead_ends from Assessment land in ResearchState.
- CHECKPOINT passes through to ANSWER in skeleton mode.
- State-transition events emit on attached hooks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.agent.loop import (
    AgentLoop,
    Assessment,
    Plan,
    PlannedCall,
    StateName,
)
from ancestors.agent.observability import HookRegistry, make_collector
from ancestors.agent.state import (
    ConfidenceLevel,
    DeadEnd,
    Fact,
    Hypothesis,
    ResearchState,
)
from ancestors.dispatch import Dispatcher, DispatchResult
from ancestors.session import bind_session, clear_session
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"


@pytest.fixture(scope="module", autouse=True)
def session():
    bind_session(load_gedcom(GEDCOM_PATH))
    yield
    clear_session()


# ---------------------------------------------------------------------------
# Test stubs: scripted Planner and Assessor.
# ---------------------------------------------------------------------------


class ScriptedPlanner:
    """A Planner that returns plans from a list in order.

    When the script is exhausted, returns an empty plan. Tracks calls for
    assertion in tests.
    """

    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.calls: list[tuple[str, ResearchState]] = []

    def __call__(self, question: str, state: ResearchState) -> Plan:
        self.calls.append((question, state))
        idx = len(self.calls) - 1
        if idx < len(self.plans):
            return self.plans[idx]
        return Plan(calls=[], rationale="(exhausted script)")


class ScriptedAssessor:
    """An Assessor that returns assessments from a list in order.

    When exhausted, returns a stuck assessment so tests don't hang.
    """

    def __init__(self, assessments: list[Assessment]) -> None:
        self.assessments = assessments
        self.calls: list[tuple[str, ResearchState, list[DispatchResult]]] = []

    def __call__(
        self,
        question: str,
        state: ResearchState,
        recent_results: list[DispatchResult],
    ) -> Assessment:
        # Copy the list — the loop clears it after this returns.
        self.calls.append((question, state, list(recent_results)))
        idx = len(self.calls) - 1
        if idx < len(self.assessments):
            return self.assessments[idx]
        return Assessment(decision="stuck", stuck_reason="assessor script exhausted")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _build_loop(
    *,
    plans: list[Plan],
    assessments: list[Assessment],
    max_dispatches: int = 50,
    stall_pattern_repeats: int = 3,
    hooks: HookRegistry | None = None,
) -> tuple[AgentLoop, ScriptedPlanner, ScriptedAssessor, Dispatcher]:
    planner = ScriptedPlanner(plans)
    assessor = ScriptedAssessor(assessments)
    dispatcher = Dispatcher(hooks=hooks)
    loop = AgentLoop(
        planner=planner,
        assessor=assessor,
        dispatcher=dispatcher,
        max_dispatches=max_dispatches,
        stall_pattern_repeats=stall_pattern_repeats,
        hooks=hooks,
    )
    return loop, planner, assessor, dispatcher


# ---------------------------------------------------------------------------
# Linear paths.
# ---------------------------------------------------------------------------


def test_single_call_then_answer():
    loop, planner, assessor, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("all_individuals", {}, "scope the corpus")],
                rationale="single lookup",
            )
        ],
        assessments=[
            Assessment(
                decision="answer",
                answer_text="There are individuals in the tree.",
                answer_confidence=ConfidenceLevel.CONFIRMED,
            )
        ],
    )
    result = loop.run("Are there people in this tree?")
    assert result.terminal_state == StateName.ANSWER
    assert result.answer_text and result.answer_text.startswith("There are")
    assert result.answer_confidence == ConfidenceLevel.CONFIRMED
    assert result.total_dispatches == 1
    assert len(planner.calls) == 1
    assert len(assessor.calls) == 1


def test_multi_call_plan_executes_all_before_assessing():
    loop, _, assessor, _ = _build_loop(
        plans=[
            Plan(
                calls=[
                    PlannedCall("all_individuals", {}, "scope"),
                    PlannedCall("all_individuals", {}, "again"),
                ],
                rationale="two calls",
            )
        ],
        assessments=[Assessment(decision="answer", answer_text="ok")],
    )
    result = loop.run("test")
    assert result.terminal_state == StateName.ANSWER
    # Assessor saw both results in one shot.
    assert len(assessor.calls[0][2]) == 2


# ---------------------------------------------------------------------------
# Branching decisions.
# ---------------------------------------------------------------------------


def test_assessor_revise_returns_to_plan():
    plans = [
        Plan(
            calls=[PlannedCall("all_individuals", {}, "first try")],
            rationale="first",
        ),
        Plan(
            calls=[PlannedCall("all_individuals", {}, "second try")],
            rationale="revised",
        ),
    ]
    assessments = [
        Assessment(decision="revise_plan", reasoning="need different approach"),
        Assessment(decision="answer", answer_text="done"),
    ]
    loop, planner, _, _ = _build_loop(plans=plans, assessments=assessments)
    result = loop.run("test")
    assert result.terminal_state == StateName.ANSWER
    assert len(planner.calls) == 2  # planner was invoked twice


def test_assessor_stuck_terminates_with_reason():
    loop, _, _, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("all_individuals", {}, "x")], rationale=""
            )
        ],
        assessments=[
            Assessment(decision="stuck", stuck_reason="no path forward")
        ],
    )
    result = loop.run("test")
    assert result.terminal_state == StateName.STUCK
    assert result.stuck_reason == "no path forward"


def test_continue_with_remaining_queue_executes_next():
    loop, _, assessor, _ = _build_loop(
        plans=[
            Plan(
                calls=[
                    PlannedCall("all_individuals", {}, "1"),
                    PlannedCall("all_individuals", {}, "2"),
                ],
                rationale="",
            )
        ],
        assessments=[Assessment(decision="answer", answer_text="done")],
    )
    result = loop.run("test")
    # Both calls dispatched; one assessor invocation after both.
    assert result.total_dispatches == 2
    assert len(assessor.calls) == 1


# ---------------------------------------------------------------------------
# Safety valves.
# ---------------------------------------------------------------------------


def test_hard_cap_forces_stuck():
    # Distinct calls per dispatch so the stall detector doesn't fire first
    # — we want to verify the hard cap specifically.
    many_calls = [
        PlannedCall(
            "get_ancestors_of",
            {"person_id": DAVID_ID, "max_generations": (i % 25) + 1},
            f"call {i}",
        )
        for i in range(20)
    ]
    loop, _, _, _ = _build_loop(
        plans=[Plan(calls=many_calls, rationale="lots")] * 20,
        assessments=[Assessment(decision="continue")] * 100,
        max_dispatches=15,
    )
    result = loop.run("test")
    assert result.terminal_state == StateName.STUCK
    assert "hard cap" in (result.stuck_reason or "").lower()
    assert result.total_dispatches == 15


def test_stall_detector_catches_repeating_pattern():
    # Same single call over and over.
    one_call = PlannedCall("all_individuals", {}, "loop")
    loop, _, _, _ = _build_loop(
        plans=[Plan(calls=[one_call], rationale="")] * 10,
        assessments=[Assessment(decision="continue")] * 10,
        stall_pattern_repeats=3,
        max_dispatches=50,
    )
    result = loop.run("test")
    assert result.terminal_state == StateName.STUCK
    assert "stall" in (result.stuck_reason or "").lower()


def test_max_state_transitions_terminates_buggy_loop():
    # A planner that returns nothing + assessor that says continue =
    # infinite ping-pong between PLAN and ASSESS. Must terminate.
    loop, _, _, _ = _build_loop(
        plans=[Plan(calls=[], rationale="empty")] * 1000,
        assessments=[Assessment(decision="continue")] * 1000,
    )
    loop.max_state_transitions = 20
    result = loop.run("test")
    assert result.terminal_state == StateName.STUCK
    assert "max state transitions" in (result.stuck_reason or "").lower()


# ---------------------------------------------------------------------------
# Visibility into errors and state.
# ---------------------------------------------------------------------------


def test_dispatch_error_visible_to_assessor():
    """Tool errors don't crash the loop; they show up in recent_results."""
    loop, _, assessor, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("does_not_exist", {}, "bad call")],
                rationale="",
            )
        ],
        assessments=[Assessment(decision="answer", answer_text="ok")],
    )
    loop.run("test")
    seen_result = assessor.calls[0][2][0]
    assert seen_result.ok is False
    assert seen_result.error.code == "unknown_tool"


def test_facts_from_assessment_applied_to_state():
    loop, _, _, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("all_individuals", {}, "")],
                rationale="",
            )
        ],
        assessments=[
            Assessment(
                decision="answer",
                answer_text="x",
                new_facts=[
                    Fact(
                        claim="There exist individuals.",
                        confidence=ConfidenceLevel.CONFIRMED,
                    )
                ],
                new_hypotheses=[
                    Hypothesis(claim="They have parents.", rationale="biological prior")
                ],
                new_dead_ends=[
                    DeadEnd(description="searched X", reason="empty result")
                ],
            )
        ],
    )
    result = loop.run("test")
    assert len(result.final_state.confirmed_facts) == 1
    assert len(result.final_state.working_hypotheses) == 1
    assert len(result.final_state.dead_ends) == 1
    # History records each application.
    assert {t.kind for t in result.final_state.history} >= {
        "add_fact",
        "add_hypothesis",
        "add_dead_end",
    }


def test_checkpoint_passes_through_to_answer_in_skeleton():
    loop, _, _, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("all_individuals", {}, "")],
                rationale="",
            )
        ],
        assessments=[
            Assessment(
                decision="checkpoint",
                answer_text="confident answer",
                answer_confidence=ConfidenceLevel.CONFIRMED,
            )
        ],
    )
    result = loop.run("test")
    assert result.terminal_state == StateName.ANSWER
    assert result.answer_text == "confident answer"


def test_checkpoint_without_answer_becomes_stuck():
    loop, _, _, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("all_individuals", {}, "")],
                rationale="",
            )
        ],
        assessments=[Assessment(decision="checkpoint")],
    )
    result = loop.run("test")
    assert result.terminal_state == StateName.STUCK
    assert "CHECKPOINT" in (result.stuck_reason or "")


# ---------------------------------------------------------------------------
# Hooks.
# ---------------------------------------------------------------------------


def test_state_transitions_emit_hook_events():
    events, observer = make_collector()
    hooks = HookRegistry()
    hooks.subscribe(observer)
    loop, _, _, _ = _build_loop(
        plans=[
            Plan(
                calls=[PlannedCall("all_individuals", {}, "")],
                rationale="",
            )
        ],
        assessments=[Assessment(decision="answer", answer_text="ok")],
        hooks=hooks,
    )
    loop.run("test")
    transitions = [e for e in events if e.kind == "on_state_transition"]
    # The exact path: INTAKE->PLAN->EXECUTE->ASSESS->ANSWER, with EXECUTE
    # possibly self-looping once. We just check the start and end.
    assert transitions[0].from_state == "INTAKE"
    assert transitions[0].to_state == "PLAN"
    assert transitions[-1].to_state == "ANSWER"


def test_run_lifecycle_events_capture_question_plan_assessment_and_answer():
    events, observer = make_collector()
    hooks = HookRegistry()
    hooks.subscribe(observer)
    loop, _, _, _ = _build_loop(
        plans=[
            Plan(
                calls=[
                    PlannedCall(
                        "all_individuals", {}, "scope the corpus first"
                    )
                ],
                rationale="want a starting set",
            )
        ],
        assessments=[
            Assessment(
                decision="answer",
                answer_text="There are 336 people in this tree.",
                answer_confidence=ConfidenceLevel.CONFIRMED,
                reasoning="Used all_individuals; size is 336.",
                new_facts=[
                    Fact(claim="corpus has 336", confidence=ConfidenceLevel.CONFIRMED)
                ],
            )
        ],
        hooks=hooks,
    )
    loop.run("How many people are in this tree?")

    by_kind = {e.kind: e for e in events}
    assert "on_run_start" in by_kind
    assert by_kind["on_run_start"].extra["question"] == "How many people are in this tree?"

    plan_events = [e for e in events if e.kind == "on_plan"]
    assert len(plan_events) == 1
    p = plan_events[0]
    assert p.extra["rationale"] == "want a starting set"
    assert p.extra["calls"][0]["tool"] == "all_individuals"
    assert p.extra["calls"][0]["justification"] == "scope the corpus first"

    assess_events = [e for e in events if e.kind == "on_assessment"]
    assert len(assess_events) == 1
    a = assess_events[0]
    assert a.extra["decision"] == "answer"
    assert a.extra["answer_text"] == "There are 336 people in this tree."
    assert a.extra["answer_confidence"] == "confirmed"
    assert a.extra["reasoning"].startswith("Used all_individuals")
    assert a.extra["facts_added"][0]["claim"] == "corpus has 336"

    complete_events = [e for e in events if e.kind == "on_run_complete"]
    assert len(complete_events) == 1
    c = complete_events[0]
    assert c.extra["terminal_state"] == "ANSWER"
    assert c.extra["answer_text"] == "There are 336 people in this tree."
    assert c.extra["confirmed_facts"][0]["claim"] == "corpus has 336"


def test_run_complete_event_captures_stuck_reason():
    events, observer = make_collector()
    hooks = HookRegistry()
    hooks.subscribe(observer)
    loop, _, _, _ = _build_loop(
        plans=[Plan(calls=[PlannedCall("all_individuals", {}, "")], rationale="")],
        assessments=[Assessment(decision="stuck", stuck_reason="no path")],
        hooks=hooks,
    )
    loop.run("impossible question")
    completes = [e for e in events if e.kind == "on_run_complete"]
    assert completes[0].extra["terminal_state"] == "STUCK"
    assert completes[0].extra["stuck_reason"] == "no path"
