"""The agent loop — explicit state machine.

States and transitions follow the original spec but with the corrections
we landed on through design conversation:

  INTAKE     -> PLAN                       (no LLM, just initialise)
  PLAN       -> EXECUTE                    (LLM produces structured Plan)
  EXECUTE    -> EXECUTE | ASSESS | STUCK   (dispatch one queued call,
                                            stop on cap or stall)
  ASSESS     -> EXECUTE | PLAN |
                CHECKPOINT | ANSWER |
                STUCK                      (LLM produces structured Assessment;
                                            loop deterministically applies
                                            state updates from it)
  CHECKPOINT -> ANSWER | STUCK             (gate on human approval; pass-through
                                            in this skeleton)
  ANSWER, STUCK                            (terminal)

Notes worth keeping in mind while reading this file:

- The Planner and Assessor are protocols. The loop does not know whether
  they are LLM-backed or scripted. Tests inject scripted ones; production
  injects LLM-backed ones built on top of `ancestors.llm`.
- ASSESS is the *only* place ResearchState gets mutated. The model produces
  judgements as Assessment fields; the loop deterministically applies them.
  This is the "no record_fact tool" decision we discussed — the model is
  out of the bookkeeping loop.
- Hard cap and stall detector are the loop's two safety valves. Neither
  has anything to do with the model's intent — they're absolute backstops.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from ancestors.agent.observability import HookRegistry
from ancestors.agent.state import (
    ConfidenceLevel,
    DeadEnd,
    Fact,
    Hypothesis,
    ResearchState,
    new_state,
)
from ancestors.dispatch import Dispatcher, DispatchResult


class StateName(StrEnum):
    INTAKE = "INTAKE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    ASSESS = "ASSESS"
    CHECKPOINT = "CHECKPOINT"
    ANSWER = "ANSWER"
    STUCK = "STUCK"


TERMINAL_STATES: frozenset[StateName] = frozenset({StateName.ANSWER, StateName.STUCK})


# ---------------------------------------------------------------------------
# Structured outputs the Planner and Assessor return.
# ---------------------------------------------------------------------------


@dataclass
class PlannedCall:
    tool: str
    args: dict
    justification: str


@dataclass
class Plan:
    calls: list[PlannedCall]
    rationale: str


@dataclass
class Assessment:
    decision: Literal["continue", "revise_plan", "checkpoint", "answer", "stuck"]
    new_facts: list[Fact] = field(default_factory=list)
    new_hypotheses: list[Hypothesis] = field(default_factory=list)
    new_dead_ends: list[DeadEnd] = field(default_factory=list)
    answer_text: str | None = None
    answer_confidence: ConfidenceLevel | None = None
    stuck_reason: str | None = None
    reasoning: str = ""


class Planner(Protocol):
    def __call__(self, question: str, state: ResearchState) -> Plan: ...


class Assessor(Protocol):
    def __call__(
        self,
        question: str,
        state: ResearchState,
        recent_results: list[DispatchResult],
    ) -> Assessment: ...


# ---------------------------------------------------------------------------
# Loop context (mutable in-place during a run) and result (returned to caller).
# ---------------------------------------------------------------------------


@dataclass
class LoopContext:
    question: str
    state: ResearchState
    queue: list[PlannedCall] = field(default_factory=list)
    recent_results: list[DispatchResult] = field(default_factory=list)
    turn_index: int = 0
    total_dispatches: int = 0
    call_history: list[str] = field(default_factory=list)
    answer_text: str | None = None
    answer_confidence: ConfidenceLevel | None = None
    stuck_reason: str | None = None
    state_name: StateName = StateName.INTAKE


@dataclass
class LoopResult:
    terminal_state: StateName
    answer_text: str | None
    answer_confidence: ConfidenceLevel | None
    stuck_reason: str | None
    final_state: ResearchState
    turns: int
    total_dispatches: int


# ---------------------------------------------------------------------------
# The loop itself.
# ---------------------------------------------------------------------------


@dataclass
class AgentLoop:
    planner: Planner
    assessor: Assessor
    dispatcher: Dispatcher
    max_dispatches: int = 50
    stall_pattern_repeats: int = 3
    max_state_transitions: int = 500
    hooks: HookRegistry | None = None

    def run(self, question: str) -> LoopResult:
        ctx = LoopContext(question=question, state=new_state(question))
        transitions = 0
        while ctx.state_name not in TERMINAL_STATES:
            self._transition(ctx)
            transitions += 1
            if transitions >= self.max_state_transitions:
                # Belt-and-braces: a buggy planner/assessor that never makes
                # forward progress mustn't loop forever.
                ctx.stuck_reason = (
                    f"Hit max state transitions ({self.max_state_transitions}) — "
                    f"likely a control-flow bug, not a model judgement."
                )
                ctx.state_name = StateName.STUCK
                self._emit_transition(StateName.STUCK, StateName.STUCK, ctx)
                break

        return LoopResult(
            terminal_state=ctx.state_name,
            answer_text=ctx.answer_text,
            answer_confidence=ctx.answer_confidence,
            stuck_reason=ctx.stuck_reason,
            final_state=ctx.state,
            turns=ctx.turn_index,
            total_dispatches=ctx.total_dispatches,
        )

    # ---- state handlers ----

    def _transition(self, ctx: LoopContext) -> None:
        old = ctx.state_name
        handler = {
            StateName.INTAKE: self._intake,
            StateName.PLAN: self._plan,
            StateName.EXECUTE: self._execute,
            StateName.ASSESS: self._assess,
            StateName.CHECKPOINT: self._checkpoint,
        }[old]
        new = handler(ctx)
        self._emit_transition(old, new, ctx)
        ctx.state_name = new

    def _intake(self, ctx: LoopContext) -> StateName:
        # Nothing to do but begin. State is already initialised by run().
        return StateName.PLAN

    def _plan(self, ctx: LoopContext) -> StateName:
        plan = self.planner(ctx.question, ctx.state)
        ctx.queue.extend(plan.calls)
        ctx.turn_index += 1
        return StateName.EXECUTE

    def _execute(self, ctx: LoopContext) -> StateName:
        if not ctx.queue:
            return StateName.ASSESS

        call = ctx.queue.pop(0)
        result = self.dispatcher.dispatch(call.tool, call.args)
        ctx.recent_results.append(result)
        ctx.total_dispatches += 1
        ctx.call_history.append(f"{call.tool}:{_args_hash(call.args)}")

        if ctx.total_dispatches >= self.max_dispatches:
            ctx.stuck_reason = f"Hit dispatch hard cap ({self.max_dispatches})."
            return StateName.STUCK
        if self._stalled(ctx):
            ctx.stuck_reason = "Loop stalled — repeated identical tool calls."
            return StateName.STUCK

        return StateName.EXECUTE if ctx.queue else StateName.ASSESS

    def _assess(self, ctx: LoopContext) -> StateName:
        assessment = self.assessor(ctx.question, ctx.state, ctx.recent_results)
        ctx.state = _apply_assessment(ctx.state, assessment)
        ctx.recent_results = []

        if assessment.decision == "answer":
            ctx.answer_text = assessment.answer_text
            ctx.answer_confidence = assessment.answer_confidence
            return StateName.ANSWER
        if assessment.decision == "stuck":
            ctx.stuck_reason = assessment.stuck_reason or "assessor reported stuck"
            return StateName.STUCK
        if assessment.decision == "checkpoint":
            ctx.answer_text = assessment.answer_text
            ctx.answer_confidence = assessment.answer_confidence
            return StateName.CHECKPOINT
        if assessment.decision == "revise_plan":
            ctx.queue.clear()
            return StateName.PLAN

        # continue
        if ctx.queue:
            return StateName.EXECUTE
        return StateName.PLAN

    def _checkpoint(self, ctx: LoopContext) -> StateName:
        # Stub behaviour: pass through. The real implementation will gate on
        # human approval. The initial trigger is "about to answer at high
        # confidence"; for now we approve and continue.
        if ctx.answer_text is None:
            ctx.stuck_reason = "Reached CHECKPOINT without a queued answer."
            return StateName.STUCK
        return StateName.ANSWER

    # ---- helpers ----

    def _stalled(self, ctx: LoopContext) -> bool:
        """Detect a tool-call pattern repeating self.stall_pattern_repeats times.

        Looks at the most recent 2*N calls; if any single (tool, args_hash)
        appears N times within that window, declares a stall. This catches
        simple loops without flagging healthy "the agent kept counting things"
        usage where ids vary.
        """
        window = max(2 * self.stall_pattern_repeats, self.stall_pattern_repeats)
        recent = ctx.call_history[-window:]
        if len(recent) < self.stall_pattern_repeats:
            return False
        return any(
            n >= self.stall_pattern_repeats for n in Counter(recent).values()
        )

    def _emit_transition(
        self, old: StateName, new: StateName, ctx: LoopContext
    ) -> None:
        if self.hooks is None:
            return
        self.hooks.emit(
            "on_state_transition",
            from_state=old.value,
            to_state=new.value,
            turn_index=ctx.turn_index,
        )


# ---------------------------------------------------------------------------
# Pure helpers (top-level so they can be unit-tested in isolation).
# ---------------------------------------------------------------------------


def _apply_assessment(state: ResearchState, a: Assessment) -> ResearchState:
    """Deterministically fold an Assessment's structured judgements into state."""
    for fact in a.new_facts:
        state = state.add_fact(fact)
    for hyp in a.new_hypotheses:
        state = state.add_hypothesis(hyp)
    for de in a.new_dead_ends:
        state = state.add_dead_end(de)
    return state


def _args_hash(args: dict) -> str:
    return hashlib.sha1(
        json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
