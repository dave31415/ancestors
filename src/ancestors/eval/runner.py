"""Run cases through a fresh agent stack and collect results.

Each case binds the supplied Corpus (so SQLite handles + dispatcher state
are isolated). The CachedAnthropic client is shared so the disk cache
survives across cases — second runs of the same question hit the cache.
"""

from __future__ import annotations

import time

from ancestors.agent.llm_agent import LlmAgent
from ancestors.agent.loop import AgentLoop, LoopResult
from ancestors.agent.observability import HookRegistry, make_collector
from ancestors.dispatch import Dispatcher
from ancestors.eval.case import Case, CaseResult, CheckResult
from ancestors.eval.corpus import Corpus
from ancestors.llm import CachedAnthropic


def run_case(
    case: Case,
    corpus: Corpus,
    llm: CachedAnthropic,
    *,
    max_dispatches: int = 30,
) -> CaseResult:
    """Execute one case against a fresh agent stack; return its CaseResult."""
    with corpus.bind():
        events, observer = make_collector()
        hooks = HookRegistry()
        hooks.subscribe(observer)
        dispatcher = Dispatcher(hooks=hooks)
        agent = LlmAgent(
            llm=llm,
            system_prompt=corpus.system_prompt(),
            dsl_tool_defs=corpus.tool_defs(),
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
