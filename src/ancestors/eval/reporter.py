"""Render eval scorecards to the terminal.

Re-uses the Style helper from trace_viewer for colour. The output is
designed to fit a normal terminal width and to make the most important
information — pass/fail status, regressions vs. baseline — visible at a
glance.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import IO

from ancestors.eval.baseline import CaseDelta
from ancestors.eval.case import CaseResult
from ancestors.trace_viewer import Style


def render_scorecard(
    results: list[CaseResult],
    *,
    deltas: list[CaseDelta] | None = None,
    style: Style | None = None,
    stream: IO[str] | None = None,
) -> None:
    style = style or Style(enabled=False)
    stream = stream or sys.stdout

    _emit_case_lines(results, style=style, stream=stream)
    _emit_summary(results, style=style, stream=stream)

    if deltas is not None:
        _emit_deltas(deltas, style=style, stream=stream)


def _emit_case_lines(
    results: list[CaseResult],
    *,
    style: Style,
    stream: IO[str],
) -> None:
    width = 72
    print(style.bold("─" * width), file=stream)
    for r in results:
        label = r.status_label
        if label == "PASS":
            tag = style.green("[PASS] ")
        elif label == "STUCK":
            tag = style.yellow("[STUCK]")
        else:
            tag = style.red("[FAIL] ")

        confidence = r.loop_result.answer_confidence
        conf_str = confidence.value if confidence is not None else "—"

        print(
            f"{tag} {r.case.fully_qualified_name:<40} "
            f"{r.dispatch_count:>3} disp  "
            f"{r.wall_seconds:>5.1f}s  "
            f"({style.confidence(conf_str)})",
            file=stream,
        )
        for c in r.failed_checks:
            print(
                f"          {style.red('✗')} {c.name}  —  {c.message}",
                file=stream,
            )


def _emit_summary(
    results: list[CaseResult],
    *,
    style: Style,
    stream: IO[str],
) -> None:
    width = 72
    print(style.bold("─" * width), file=stream)
    counts = Counter(r.status_label for r in results)
    n = len(results)
    n_pass = counts.get("PASS", 0)
    n_fail = counts.get("FAIL", 0)
    n_stuck = counts.get("STUCK", 0)
    total_dispatches = sum(r.dispatch_count for r in results)
    total_llm = sum(r.llm_call_count for r in results)
    total_time = sum(r.wall_seconds for r in results)

    pct = (n_pass / n * 100) if n else 0.0
    print(style.bold("Summary"), file=stream)
    print(
        f"  {n_pass}/{n} passed ({pct:.0f}%)  "
        f"|  {style.red(str(n_fail) + ' FAIL') if n_fail else '0 FAIL'}  "
        f"|  {style.yellow(str(n_stuck) + ' STUCK') if n_stuck else '0 STUCK'}",
        file=stream,
    )
    print(
        f"  totals: {total_dispatches} dispatches, "
        f"{total_llm} LLM calls, "
        f"{total_time:.1f}s wall time",
        file=stream,
    )


def _emit_deltas(
    deltas: list[CaseDelta],
    *,
    style: Style,
    stream: IO[str],
) -> None:
    width = 72
    print(style.bold("─" * width), file=stream)
    print(style.bold("vs. baseline"), file=stream)

    changes = [d for d in deltas if d.kind != "unchanged"]
    if not changes:
        print(f"  {style.dim('no changes from baseline')}", file=stream)
        return

    # Group by kind for readability
    by_kind: dict[str, list[CaseDelta]] = {}
    for d in changes:
        by_kind.setdefault(d.kind, []).append(d)

    headers = {
        "status_change": ("status changes", style.yellow),
        "efficiency_change": ("efficiency drift", style.yellow),
        "new": ("new cases", style.cyan),
        "removed": ("removed cases", style.dim),
    }
    for kind, (header, colour) in headers.items():
        items = by_kind.get(kind)
        if not items:
            continue
        print(f"  {colour(header)}:", file=stream)
        for d in items:
            print(f"    - {d.case_name}: {d.detail}", file=stream)
