"""Tests for the LLM-backed Planner and Assessor.

We don't call the real API. We swap CachedAnthropic.messages with a stub
that returns crafted Message objects, then verify:

- plan() extracts tool_use blocks into PlannedCalls.
- assess() extracts a submit_assessment tool_use's input into Assessment.
- Robustness fallbacks fire when the model returns the wrong shape.
- The DispatchResult formatter renders set/value/error results compactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from ancestors.agent.llm_agent import (
    LlmAgent,
    SubmitAssessmentInput,
    submit_assessment_tool_def,
)
from ancestors.agent.state import (
    ConfidenceLevel,
    DeadEnd,
    Fact,
    Hypothesis,
    new_state,
)
from ancestors.dispatch import DispatchResult
from ancestors.llm import CachedAnthropic
from ancestors.models import ToolError


def _msg(blocks: list[Any], *, stop_reason: str = "tool_use") -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        content=blocks,
        model="claude-opus-4-7",
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=10,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _tool_use(name: str, input_: dict) -> ToolUseBlock:
    return ToolUseBlock(id=f"toolu_{name}", type="tool_use", name=name, input=input_)


def _text(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


@pytest.fixture
def agent(tmp_path: Path) -> tuple[LlmAgent, MagicMock]:
    llm = CachedAnthropic(api_key="fake", cache_dir=tmp_path / "cache", use_cache=True)
    llm.client = MagicMock()
    a = LlmAgent(
        llm=llm,
        system_prompt="(test system prompt)",
        dsl_tool_defs=[{"name": "fake", "description": "x", "input_schema": {"type": "object"}}],
    )
    return a, llm.client


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


def test_plan_extracts_tool_use_blocks(agent):
    a, client = agent
    client.messages.create.return_value = _msg(
        [
            _text("I'll start by scoping the corpus."),
            _tool_use("all_individuals", {}),
            _tool_use("get_ancestors_of", {"person_id": "@I123@", "max_generations": 5}),
        ]
    )
    plan = a.plan("test question", new_state("test question"))
    assert [c.tool for c in plan.calls] == ["all_individuals", "get_ancestors_of"]
    assert plan.calls[1].args == {"person_id": "@I123@", "max_generations": 5}
    assert "scoping" in plan.rationale


def test_plan_no_tool_use_returns_empty_plan(agent):
    a, client = agent
    client.messages.create.return_value = _msg(
        [_text("I think I can answer from state alone.")],
        stop_reason="end_turn",
    )
    plan = a.plan("test", new_state("test"))
    assert plan.calls == []
    assert "answer from state" in plan.rationale


# ---------------------------------------------------------------------------
# assess()
# ---------------------------------------------------------------------------


def test_assess_extracts_assessment_input(agent):
    a, client = agent
    client.messages.create.return_value = _msg(
        [
            _tool_use(
                "submit_assessment",
                {
                    "decision": "answer",
                    "reasoning": "The corpus contains 336 individuals.",
                    "new_facts": [
                        {
                            "claim": "There are 336 individuals in the tree.",
                            "confidence": "confirmed",
                            "sources": ["tool:all_individuals"],
                            "rationale": None,
                        }
                    ],
                    "new_hypotheses": [],
                    "new_dead_ends": [],
                    "answer_text": "The tree contains 336 individuals.",
                    "answer_confidence": "confirmed",
                    "stuck_reason": None,
                },
            )
        ]
    )
    assessment = a.assess("How many?", new_state("How many?"), [])
    assert assessment.decision == "answer"
    assert assessment.answer_text == "The tree contains 336 individuals."
    assert assessment.answer_confidence == ConfidenceLevel.CONFIRMED
    assert assessment.new_facts[0].claim.startswith("There are 336")


def test_assess_falls_back_to_stuck_if_no_submit_assessment(agent):
    a, client = agent
    client.messages.create.return_value = _msg(
        [_text("I forgot to call the tool")],
        stop_reason="end_turn",
    )
    assessment = a.assess("test", new_state("test"), [])
    assert assessment.decision == "stuck"
    assert "did not call submit_assessment" in (assessment.stuck_reason or "")


def test_assess_falls_back_on_invalid_input(agent):
    a, client = agent
    # decision is not a valid literal.
    client.messages.create.return_value = _msg(
        [_tool_use("submit_assessment", {"decision": "not_a_real_decision"})]
    )
    assessment = a.assess("test", new_state("test"), [])
    assert assessment.decision == "stuck"
    assert "failed validation" in (assessment.stuck_reason or "")


# ---------------------------------------------------------------------------
# DispatchResult formatting in the ASSESS prompt
# ---------------------------------------------------------------------------


def test_format_results_renders_sets_values_and_errors(agent):
    a, _ = agent
    results = [
        DispatchResult(ok=True, tool="all_individuals", set_handle="h_1", set_size=336),
        DispatchResult(ok=True, tool="count", value=336),
        DispatchResult(
            ok=False,
            tool="bad_tool",
            error=ToolError(code="unknown_tool", message="Tool 'bad_tool' is not registered."),
        ),
    ]
    text = a._format_results(results)
    assert "IdSet h_1 size=336" in text
    assert "336" in text  # the scalar value appears, possibly on its own line
    assert "unknown_tool" in text


def test_format_results_empty(agent):
    a, _ = agent
    assert "no tool calls" in a._format_results([])


# ---------------------------------------------------------------------------
# Tool definition for submit_assessment is well-formed.
# ---------------------------------------------------------------------------


def test_submit_assessment_tool_def_is_anthropic_compatible():
    td = submit_assessment_tool_def()
    assert td["name"] == "submit_assessment"
    assert td["input_schema"]["type"] == "object"
    assert "decision" in td["input_schema"]["properties"]
    # No dangling refs in the schema (would error at the API).
    assert "$defs" not in td["input_schema"]


def test_submit_assessment_input_accepts_minimal_continue():
    # The most common case: model says "continue", no facts yet.
    parsed = SubmitAssessmentInput.model_validate({"decision": "continue"})
    assert parsed.decision == "continue"
    assert parsed.new_facts == []
