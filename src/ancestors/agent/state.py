"""ResearchState — the agent's working memory across a single question.

The state is what the agent knows, suspects, has ruled out, and still wants
to find out. It is the substrate the ASSESS step reads and updates.

Design choices worth flagging:

- Updates return new ResearchState instances. Pydantic's `model_copy(update=…)`
  gives us this cheaply. Immutability means a trace can replay any past state
  by reading the history log, and a buggy update can't corrupt prior beliefs.

- No `record_fact` tool exposed to the model. ASSESS is the only writer.
  The model produces a *judgement* in its structured response and the loop
  code applies that judgement to the state. This keeps the model out of the
  bookkeeping loop where it would have the most opportunity to forget,
  duplicate, or misrecord.

- A history log records every transition. Useful for the trace and for
  audits where the question "when did we believe X?" matters.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field


class ConfidenceLevel(StrEnum):
    """Genealogical Proof Standard confidence vocabulary, ordered."""

    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    SPECULATIVE = "speculative"
    REFUTED = "refuted"


class Fact(BaseModel):
    """A claim believed to be true, with evidence trail."""

    claim: str
    confidence: ConfidenceLevel
    sources: list[str] = Field(default_factory=list)
    rationale: str | None = None


class Hypothesis(BaseModel):
    """A claim under investigation but not yet established."""

    claim: str
    rationale: str
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.POSSIBLE


class DeadEnd(BaseModel):
    """A search path that produced no progress.

    Recording these explicitly keeps the agent from repeating fruitless
    searches and gives the human reviewer the audit trail the spec's
    'Reasonably Exhaustive Search' principle requires.
    """

    description: str
    reason: str


class StateTransition(BaseModel):
    """A single update appended to the history log."""

    kind: str
    detail: dict[str, str | int | float] = Field(default_factory=dict)


class ResearchState(BaseModel):
    """The agent's working memory for one research question."""

    question: str
    confirmed_facts: list[Fact] = Field(default_factory=list)
    working_hypotheses: list[Hypothesis] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    dead_ends: list[DeadEnd] = Field(default_factory=list)
    history: list[StateTransition] = Field(default_factory=list)

    # ---- transitions: each returns a NEW ResearchState ----

    def add_fact(self, fact: Fact) -> Self:
        return self.model_copy(
            update={
                "confirmed_facts": [*self.confirmed_facts, fact],
                "history": [*self.history, StateTransition(
                    kind="add_fact",
                    detail={"claim": fact.claim, "confidence": fact.confidence},
                )],
            }
        )

    def add_hypothesis(self, hypothesis: Hypothesis) -> Self:
        return self.model_copy(
            update={
                "working_hypotheses": [*self.working_hypotheses, hypothesis],
                "history": [*self.history, StateTransition(
                    kind="add_hypothesis",
                    detail={"claim": hypothesis.claim},
                )],
            }
        )

    def promote_hypothesis(self, index: int, confidence: ConfidenceLevel) -> Self:
        """Move a working hypothesis into confirmed_facts."""
        hyp = self.working_hypotheses[index]
        remaining = [h for i, h in enumerate(self.working_hypotheses) if i != index]
        fact = Fact(
            claim=hyp.claim,
            confidence=confidence,
            sources=hyp.supporting,
            rationale=hyp.rationale,
        )
        return self.model_copy(
            update={
                "working_hypotheses": remaining,
                "confirmed_facts": [*self.confirmed_facts, fact],
                "history": [*self.history, StateTransition(
                    kind="promote_hypothesis",
                    detail={"claim": hyp.claim, "confidence": confidence},
                )],
            }
        )

    def add_open_question(self, question: str) -> Self:
        return self.model_copy(
            update={
                "open_questions": [*self.open_questions, question],
                "history": [*self.history, StateTransition(
                    kind="add_open_question",
                    detail={"question": question},
                )],
            }
        )

    def close_open_question(self, index: int) -> Self:
        remaining = [q for i, q in enumerate(self.open_questions) if i != index]
        closed = self.open_questions[index]
        return self.model_copy(
            update={
                "open_questions": remaining,
                "history": [*self.history, StateTransition(
                    kind="close_open_question",
                    detail={"question": closed},
                )],
            }
        )

    def add_dead_end(self, dead_end: DeadEnd) -> Self:
        return self.model_copy(
            update={
                "dead_ends": [*self.dead_ends, dead_end],
                "history": [*self.history, StateTransition(
                    kind="add_dead_end",
                    detail={"description": dead_end.description},
                )],
            }
        )

    # ---- readers ----

    def summary(self) -> str:
        """Compact textual summary suitable for an LLM prompt."""
        return (
            f"Question: {self.question}\n"
            f"Confirmed facts: {len(self.confirmed_facts)}\n"
            f"Working hypotheses: {len(self.working_hypotheses)}\n"
            f"Open questions: {len(self.open_questions)}\n"
            f"Dead ends: {len(self.dead_ends)}\n"
        )


def new_state(question: str) -> ResearchState:
    """Construct a fresh state for a question."""
    return ResearchState(question=question)
