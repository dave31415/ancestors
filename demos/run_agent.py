"""End-to-end demo: ask a real question, watch the agent answer it.

Loads the GEDCOM, binds a session, wires the dispatcher + observer + LLM
agent + loop, then asks one question and prints the result. The observer
prints each tool call and state transition as they happen, so you can watch
the reasoning unfold.

Run with:
    uv run python demos/run_agent.py

Requires ANTHROPIC_API_KEY in env or ~/.config/ancestors/keys/anthropic.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ancestors.agent.llm_agent import LlmAgent
from ancestors.agent.loop import AgentLoop
from ancestors.agent.observability import (
    HookRegistry,
    JsonlTraceWriter,
    TraceEvent,
)
from ancestors.agent.prompts import build_system_prompt
from ancestors.agent.schema import export_tools_for_anthropic
from ancestors.dispatch import Dispatcher
from ancestors.llm import CachedAnthropic
from ancestors.session import bind_session
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).resolve().parent.parent / "data" / "export-Ancestors.ged"
DEFAULT_QUESTION = (
    "Who is David Johnston's paternal grandfather, and where was he born?"
)


def print_event(event: TraceEvent) -> None:
    """Stream a one-line summary of each event to stdout as it fires."""
    if event.kind == "on_state_transition":
        print(f"  [state] {event.from_state} → {event.to_state}", file=sys.stderr)
    elif event.kind == "before_dispatch":
        args_short = str(event.args)
        if len(args_short) > 90:
            args_short = args_short[:87] + "..."
        print(f"  [tool ] {event.tool}({args_short})", file=sys.stderr)
    elif event.kind == "after_dispatch":
        rs = event.result_summary or {}
        if "set_handle" in rs:
            print(
                f"  [ok   ] → {rs['set_handle']} size={rs['set_size']} "
                f"({event.duration_ms:.0f}ms)",
                file=sys.stderr,
            )
        else:
            print(
                f"  [ok   ] → {rs} ({event.duration_ms:.0f}ms)",
                file=sys.stderr,
            )
    elif event.kind == "on_error":
        print(
            f"  [err  ] {event.error_code}: {event.error_message}",
            file=sys.stderr,
        )


def main(question: str = DEFAULT_QUESTION) -> None:
    print(f"Loading GEDCOM from {GEDCOM_PATH}...", file=sys.stderr)
    db = load_gedcom(GEDCOM_PATH)
    bind_session(db)

    hooks = HookRegistry()
    hooks.subscribe(print_event)
    hooks.subscribe(JsonlTraceWriter())

    dispatcher = Dispatcher(hooks=hooks)
    llm = CachedAnthropic()
    agent = LlmAgent(
        llm=llm,
        system_prompt=build_system_prompt(db),
        dsl_tool_defs=export_tools_for_anthropic(),
    )
    loop = AgentLoop(
        planner=agent.plan,
        assessor=agent.assess,
        dispatcher=dispatcher,
        hooks=hooks,
        max_dispatches=30,
    )

    print(f"\nQuestion: {question}\n", file=sys.stderr)
    print(f"Session: {hooks.session_id}\n", file=sys.stderr)

    result = loop.run(question)

    print()
    print("=" * 72)
    print(f"Terminal state: {result.terminal_state.value}")
    print(f"Turns: {result.turns}   Dispatches: {result.total_dispatches}")
    if result.answer_text:
        print(f"Confidence: {result.answer_confidence}")
        print()
        print("Answer:")
        print(result.answer_text)
    if result.stuck_reason:
        print(f"Stuck reason: {result.stuck_reason}")
    print("=" * 72)

    print()
    print(f"Confirmed facts ({len(result.final_state.confirmed_facts)}):")
    for f in result.final_state.confirmed_facts:
        print(f"  - [{f.confidence}] {f.claim}")
    if result.final_state.working_hypotheses:
        print(f"Working hypotheses ({len(result.final_state.working_hypotheses)}):")
        for h in result.final_state.working_hypotheses:
            print(f"  - [{h.confidence}] {h.claim}")
    if result.final_state.dead_ends:
        print(f"Dead ends ({len(result.final_state.dead_ends)}):")
        for d in result.final_state.dead_ends:
            print(f"  - {d.description}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    main(q)
