"""LLM-backed Planner and Assessor implementations.

Wires the Anthropic `messages.create` call into the agent loop's protocols.
Two methods on one class so they share configuration:

  .plan   — implements Planner. Sends the question and current ResearchState
            summary; returns the model's tool_use blocks as PlannedCalls.
            Uses the real DSL tools in the API's `tools` parameter; lets the
            model choose (tool_choice="auto") which to call and with what
            arguments.

  .assess — implements Assessor. Sends the question, current state, and a
            formatted view of the last batch of DispatchResults; forces the
            model to emit one `submit_assessment` tool_use whose input
            satisfies SubmitAssessmentInput. The loop then deterministically
            folds that structured output into ResearchState.

Both calls are stateless from the Anthropic side — we don't carry the
conversation across PLAN/ASSESS turns. The ResearchState summary in the
prompt is how cross-turn memory flows. This keeps the implementation simple
and means turn N+1 doesn't depend on the exact phrasing of turn N's model
output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ancestors.agent.loop import Assessment, Plan, PlannedCall
from ancestors.agent.schema import _normalize_schema
from ancestors.agent.state import (
    ConfidenceLevel,
    DeadEnd,
    Fact,
    Hypothesis,
    ResearchState,
)
from ancestors.dispatch import DispatchResult, Dispatcher
from ancestors.llm import CachedAnthropic

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema for ASSESS.
#
# The `submit_assessment` meta-tool exists only to give the model a typed
# slot to drop its analysis into. The dispatcher never sees it; the loop
# never executes it; it lives entirely in the LLM-to-agent boundary as a
# structured-output mechanism.
# ---------------------------------------------------------------------------


class SubmitAssessmentInput(BaseModel):
    decision: Literal["continue", "revise_plan", "checkpoint", "answer", "stuck"] = (
        Field(
            description=(
                "Where to take the loop next. 'continue' to dispatch more queued "
                "calls or replan; 'revise_plan' to discard the current plan and "
                "make a new one; 'checkpoint' if you intend to commit to a "
                "high-confidence answer (human gate); 'answer' to produce the "
                "final answer; 'stuck' if no further progress is possible."
            )
        )
    )
    reasoning: str = Field(
        default="",
        description="Short justification of the decision, for the trace.",
    )
    new_facts: list[Fact] = Field(
        default_factory=list,
        description=(
            "Claims newly believed at the stated confidence, with sources. "
            "Only add a fact when the evidence supports it."
        ),
    )
    new_hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Plausible claims under investigation but not established.",
    )
    new_dead_ends: list[DeadEnd] = Field(
        default_factory=list,
        description=(
            "Search paths that produced no progress. Recording them prevents "
            "repeating fruitless searches."
        ),
    )
    answer_text: str | None = Field(
        default=None,
        description=(
            "Final answer text, present iff decision is 'answer' or "
            "'checkpoint'. Should state the conclusion concisely."
        ),
    )
    answer_confidence: ConfidenceLevel | None = Field(
        default=None,
        description="Calibrated confidence level for the answer.",
    )
    stuck_reason: str | None = Field(
        default=None,
        description="Why the loop is stuck, present iff decision is 'stuck'.",
    )


SUBMIT_ASSESSMENT_DESCRIPTION = (
    "Submit your structured analysis of the latest tool results. Use this "
    "to record new facts/hypotheses/dead ends, direct the loop's next step, "
    "and (if ready) produce the final answer or report being stuck."
)


def submit_assessment_tool_def() -> dict[str, Any]:
    """Anthropic tool definition for the assessment meta-tool."""
    raw = SubmitAssessmentInput.model_json_schema()
    return {
        "name": "submit_assessment",
        "description": SUBMIT_ASSESSMENT_DESCRIPTION,
        "input_schema": _normalize_schema(raw),
    }


# ---------------------------------------------------------------------------
# LlmAgent: bundles the Planner and Assessor implementations.
# ---------------------------------------------------------------------------


@dataclass
class LlmAgent:
    llm: CachedAnthropic
    system_prompt: str
    dsl_tool_defs: list[dict[str, Any]]
    model: str | None = None
    # Optional. When provided, PLAN/ASSESS prompts include a summary of
    # the current IdSet handles so the model reuses h_1 rather than calling
    # the same producer again. Combined with dispatcher memoization, this is
    # the load-bearing fix for the repeat-call pathology.
    dispatcher: Dispatcher | None = None

    # ---- Planner ----

    def plan(self, question: str, state: ResearchState) -> Plan:
        prompt = self._build_plan_prompt(question, state)
        response = self.llm.messages(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            tools=self.dsl_tool_defs,
            model=self.model,
        )
        return self._parse_plan(response)

    # ---- Assessor ----

    def assess(
        self,
        question: str,
        state: ResearchState,
        recent_results: list[DispatchResult],
    ) -> Assessment:
        prompt = self._build_assess_prompt(question, state, recent_results)
        response = self.llm.messages(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            tools=[submit_assessment_tool_def()],
            tool_choice={"type": "tool", "name": "submit_assessment"},
            model=self.model,
        )
        return self._parse_assessment(response)

    # ---- Prompt construction ----

    def _build_plan_prompt(self, question: str, state: ResearchState) -> str:
        return (
            f"Research question: {question}\n\n"
            f"Current research state:\n{state.summary()}\n\n"
            f"{self._handles_block()}"
            "Plan the next concrete step toward answering. Call the tools "
            "you need — each call must move the investigation forward. "
            "Reuse existing handles when they fit; do NOT re-call a "
            "producer if the handle you need is already listed above. "
            "Aim for 1–4 tool calls. If the prior turn's results already "
            "contain the answer, emit no tool calls so the assessor can "
            "produce the answer directly."
        )

    def _build_assess_prompt(
        self,
        question: str,
        state: ResearchState,
        recent_results: list[DispatchResult],
    ) -> str:
        results_text = self._format_results(recent_results)
        return (
            f"Research question: {question}\n\n"
            f"Current research state:\n{state.summary()}\n\n"
            f"{self._handles_block()}"
            f"Results from your last batch of tool calls:\n{results_text}\n\n"
            "Now produce a structured assessment via `submit_assessment`. "
            "Specifically:\n\n"
            "1. **Record what you learned.** Every meaningful result should "
            "become a fact, hypothesis, or dead end. Set handles by "
            "themselves are not facts — interpret what the set represents "
            "(e.g. 'h_2 size=4 is the set of people surnamed Johnston'). "
            "Without recorded notes, next turn you will not remember what "
            "h_2 was.\n\n"
            "2. **Decide the next step.** Choose `continue` if you need "
            "more data, `revise_plan` if your approach was wrong, `answer` "
            "if you have enough to conclude, or `stuck` if no path forward "
            "exists. Do not choose `continue` without having recorded at "
            "least one fact, hypothesis, or dead end from this turn — "
            "otherwise the next plan turn will start blind and the loop "
            "will spin.\n\n"
            "3. **Be honest about confidence.** Use the calibration vocabulary "
            "from the system prompt. Unsupported claims should be hypotheses, "
            "not facts."
        )

    def _handles_block(self) -> str:
        """Render the dispatcher's handle inventory for inclusion in prompts.

        Empty string when no dispatcher is attached or no handles exist —
        keeps the prompt clean when there's nothing to say.
        """
        if self.dispatcher is None:
            return ""
        summary = self.dispatcher.handles_summary()
        if "no IdSet handles" in summary:
            return ""
        return f"Known IdSet handles in this session:\n{summary}\n\n"

    def _format_results(self, results: list[DispatchResult]) -> str:
        """A model-readable summary of dispatch results.

        Sets are summarised as handle+size (the agent never sees raw ids).
        Scalars are shown verbatim. Hydrated record lists are rendered with
        the fields a researcher would actually want — name, sex, key dates
        and places, ids of related records. Notes are truncated to keep the
        prompt tractable.

        Without enough detail here the model can't make decisions about
        what it just saw, and will tend to re-call the same tool hoping to
        get more — which the stall detector then catches as a loop.
        """
        if not results:
            return "(no tool calls were executed this turn)"
        lines = []
        for i, r in enumerate(results, 1):
            head = f"{i}. {r.tool}"
            if not r.ok and r.error is not None:
                lines.append(f"{head} — ERROR {r.error.code}: {r.error.message}")
                continue
            if r.set_handle is not None:
                lines.append(f"{head} → IdSet {r.set_handle} size={r.set_size}")
                continue
            lines.append(f"{head} →")
            lines.append(_format_value(r.value, indent="    "))
        return "\n".join(lines)

    # ---- Response parsing ----

    def _parse_plan(self, response: Any) -> Plan:
        calls: list[PlannedCall] = []
        rationale_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                calls.append(
                    PlannedCall(
                        tool=block.name,
                        args=dict(block.input or {}),
                        justification="",
                    )
                )
            elif getattr(block, "type", None) == "text":
                rationale_parts.append(block.text)
        return Plan(calls=calls, rationale=" ".join(rationale_parts).strip())

    def _parse_assessment(self, response: Any) -> Assessment:
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "submit_assessment"
            ):
                return _assessment_from_input(block.input or {})
        return Assessment(
            decision="stuck",
            stuck_reason="Model did not call submit_assessment in ASSESS turn.",
        )


def _assessment_from_input(input_dict: dict[str, Any]) -> Assessment:
    # Defensive against a Claude regression where the model writes
    # XML-style tags inside the JSON string values of tool input. See
    # _strip_embedded_xml_fields for details.
    input_dict = _strip_embedded_xml_fields(input_dict)
    try:
        parsed = SubmitAssessmentInput.model_validate(input_dict)
    except ValidationError as exc:
        log.warning("submit_assessment validation failed: %s", exc)
        return Assessment(
            decision="stuck",
            stuck_reason=f"submit_assessment input failed validation: {exc}",
        )
    return Assessment(
        decision=parsed.decision,
        new_facts=parsed.new_facts,
        new_hypotheses=parsed.new_hypotheses,
        new_dead_ends=parsed.new_dead_ends,
        answer_text=parsed.answer_text,
        answer_confidence=parsed.answer_confidence,
        stuck_reason=parsed.stuck_reason,
        reasoning=parsed.reasoning,
    )


# Regex matches a trailing fragment like:
#   </field_name>\n<parameter name="next_field">next_value
# possibly followed by another such pair, with optional closing </parameter>.
_XML_LEAK_RE = re.compile(
    r"</(?P<this_field>[a-zA-Z_]\w*)>\s*"
    r"(?:<parameter\s+name=\"(?P<next_field>[a-zA-Z_]\w*)\">)?"
    r"(?P<next_value>.*?)"
    r"(?:</parameter>)?\s*$",
    re.DOTALL,
)


def _strip_embedded_xml_fields(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Recover from Claude's occasional XML-tag-in-JSON leak.

    The submit_assessment tool expects a JSON object whose fields are
    populated as separate keys. Occasionally the model regresses to the
    XML-tagged tool-use idiom and writes:

        answer_text = "...real answer.</answer_text>
                       <parameter name=\"answer_confidence\">probable"

    leaving the actual answer_confidence field unset. This helper detects
    the trailing tags, trims them from the originating field, and copies
    the embedded values into their intended fields. Iterates until no
    embedded fields remain so chained leaks ("answer_text → confidence →
    stuck_reason → …") all get rescued.
    """
    cleaned = dict(input_dict)
    text_fields = ("answer_text", "stuck_reason", "reasoning")
    changed = True
    while changed:
        changed = False
        for field_name in text_fields:
            value = cleaned.get(field_name)
            if not isinstance(value, str):
                continue
            match = _XML_LEAK_RE.search(value)
            if not match or match.group("this_field") != field_name:
                continue
            cleaned[field_name] = value[: match.start()].rstrip()
            next_field = match.group("next_field")
            next_value = (match.group("next_value") or "").strip()
            if next_field and next_value and next_field not in cleaned:
                cleaned[next_field] = next_value
                changed = True
            elif not next_field:
                # Trailing close tag with no follow-on field — done.
                changed = True
    return cleaned


def _format_value(value: Any, indent: str = "") -> str:
    """Render a tool result value as model-readable text.

    Lists of Individual-shaped records are special-cased to highlight the
    fields most useful for the agent's downstream reasoning. Other values
    fall through to JSON, with note fields truncated.
    """
    if value is None:
        return f"{indent}(null)"
    if isinstance(value, (int, float, bool, str)):
        s = json.dumps(value)
        return f"{indent}{s if len(s) < 800 else s[:800] + '…'}"
    if isinstance(value, list):
        if not value:
            return f"{indent}(empty list)"
        # Heuristic: if the first element looks like an Individual record
        # (has id + names), render each one compactly.
        if isinstance(value[0], dict) and "id" in value[0] and "names" in value[0]:
            return "\n".join(
                _format_individual(ind, indent) for ind in value
            )
        # Otherwise dump JSON with truncation.
        return _truncate(json.dumps(value, indent=2, default=str), indent)
    if isinstance(value, dict):
        return _truncate(json.dumps(value, indent=2, default=str), indent)
    return f"{indent}{type(value).__name__}"


def _format_individual(ind: dict[str, Any], indent: str) -> str:
    """One-shot rendering of an Individual record for the model.

    Surfaces id, primary name, sex, birth/death/burial dates and places,
    family links, and source ids. Notes are truncated hard — the model can
    request a specific note via a future tool if it ever needs to.
    """
    name = "?"
    if ind.get("names"):
        name = ind["names"][0].get("full") or "?"
    parts = [f"{indent}- id={ind.get('id')!r} name={name!r} sex={ind.get('sex')}"]
    for ev_key in ("birth", "death", "burial"):
        ev = ind.get(ev_key)
        if not ev:
            continue
        date = (ev.get("date") or {}).get("raw")
        place = ev.get("place") or {}
        place_str = " / ".join(
            p for p in (place.get("city"), place.get("state"), place.get("country"), place.get("raw"))
            if p
        )
        sources = ev.get("source_ids") or []
        line = f"{indent}    {ev_key}: date={date!r} place={place_str!r}"
        if sources:
            line += f" sources={sources}"
        parts.append(line)
    if ind.get("families_as_child"):
        parts.append(f"{indent}    families_as_child={ind['families_as_child']}")
    if ind.get("families_as_spouse"):
        parts.append(f"{indent}    families_as_spouse={ind['families_as_spouse']}")
    notes = ind.get("notes") or []
    if notes:
        first = (notes[0] or "")[:200]
        more = f" (+{len(notes) - 1} more)" if len(notes) > 1 else ""
        parts.append(f"{indent}    notes: {first!r}{more}")
    return "\n".join(parts)


def _truncate(text: str, indent: str, max_chars: int = 2000) -> str:
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… (truncated)"
    return "\n".join(indent + line for line in text.splitlines())
