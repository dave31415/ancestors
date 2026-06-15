"""Run cases through a fresh agent stack and collect results.

Each case gets its own Session (so SQLite handles + dispatcher state are
isolated). The CachedAnthropic client is shared so the disk cache survives
across cases — second runs of the same question hit the cache.
"""

from __future__ import annotations

import time
from pathlib import Path

from ancestors.agent.llm_agent import LlmAgent
from ancestors.agent.loop import AgentLoop, LoopResult
from ancestors.agent.observability import HookRegistry, TraceEvent, make_collector
from ancestors.agent.prompts import build_system_prompt
from ancestors.agent.schema import export_tools_for_anthropic
from ancestors.dispatch import Dispatcher
from ancestors.eval.case import Case, CaseResult, CheckResult
from ancestors.llm import CachedAnthropic
from ancestors.models import GedcomDatabase
from ancestors.session import bind_session, clear_session
from ancestors.tools.gedcom import load_gedcom


def load_corpus(gedcom_path: Path) -> GedcomDatabase:
    """Load the GEDCOM once and reuse for every case."""
    return load_gedcom(gedcom_path)


def run_case(
    case: Case,
    db: GedcomDatabase,
    llm: CachedAnthropic,
    *,
    max_dispatches: int = 30,
) -> CaseResult:
    """Execute one case against a fresh agent stack; return its CaseResult."""
    bind_session(db)
    try:
        events, observer = make_collector()
        hooks = HookRegistry()
        hooks.subscribe(observer)
        dispatcher = Dispatcher(hooks=hooks)
        agent = LlmAgent(
            llm=llm,
            system_prompt=build_system_prompt(db),
            dsl_tool_defs=export_tools_for_anthropic(),
            dispatcher=dispatcher,
        )
        loop = AgentLoop(
            planner=agent.plan,
            assessor=agent.assess,
            dispatcher=dispatcher,
            hooks=hooks,
            max_dispatches=max_dispatches,
        )

        start = time.time()
        loop_result: LoopResult = loop.run(case.question)
        wall_seconds = time.time() - start
    finally:
        clear_session()

    check_results = [_run_one_check(c, loop_result, events) for c in case.checks]
    dispatch_count = loop_result.total_dispatches
    llm_call_count = sum(
        1 for e in events if e.kind in ("on_plan", "on_assessment")
    )
    tools_used = sorted({
        e.tool for e in events
        if e.kind == "before_dispatch" and e.tool
    })

    return CaseResult(
        case=case,
        loop_result=loop_result,
        check_results=check_results,
        wall_seconds=wall_seconds,
        dispatch_count=dispatch_count,
        llm_call_count=llm_call_count,
        tools_used=tools_used,
    )


def _run_one_check(check, loop_result, events) -> CheckResult:
    try:
        return check(loop_result, events)
    except Exception as exc:  # noqa: BLE001 — check failures must not crash the run
        return CheckResult(
            passed=False,
            message=f"check raised: {type(exc).__name__}: {exc}",
            name=getattr(check, "__name__", "<unknown>"),
        )
