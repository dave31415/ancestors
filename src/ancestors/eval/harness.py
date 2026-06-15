"""Generic eval harness entry point.

Domain-agnostic: takes a Corpus, runs the selected cases, prints the
scorecard, optionally blesses a baseline. The domain CLI is responsible
for constructing the Corpus and parsing its own flags (e.g. --gedcom).
"""

from __future__ import annotations

import sys
from pathlib import Path

from ancestors.eval.baseline import (
    DEFAULT_BASELINE_PATH,
    diff_against_baseline,
    load_baseline,
    save_baseline,
)
from ancestors.eval.case import Case
from ancestors.eval.cases import all_cases, case_by_name, cases_in_suite
from ancestors.eval.corpus import Corpus
from ancestors.eval.reporter import render_scorecard
from ancestors.eval.runner import run_case
from ancestors.llm import CachedAnthropic
from ancestors.trace_viewer import Style


def select_cases(
    *, suite: str | None = None, case: str | None = None
) -> tuple[list[Case], str | None]:
    """Resolve case selection. Returns (cases, error_message)."""
    if case:
        c = case_by_name(case)
        if c is None:
            return [], f"error: case {case!r} not found"
        return [c], None
    if suite:
        cases = cases_in_suite(suite)
        if not cases:
            return [], f"error: no cases in suite {suite!r}"
        return cases, None
    return all_cases(), None


def run_eval(
    corpus: Corpus,
    *,
    suite: str | None = None,
    case: str | None = None,
    no_cache: bool = False,
    no_color: bool = False,
    bless: bool = False,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
) -> int:
    """Run the harness against a corpus. Returns a shell exit code."""
    style = Style(enabled=not no_color and sys.stdout.isatty())

    cases, err = select_cases(suite=suite, case=case)
    if err is not None:
        print(err, file=sys.stderr)
        return 1

    llm = CachedAnthropic(use_cache=not no_cache)

    results = []
    for i, c in enumerate(cases, 1):
        print(
            f"[{i}/{len(cases)}] {c.fully_qualified_name}…",
            file=sys.stderr,
            flush=True,
        )
        result = run_case(c, corpus, llm)
        results.append(result)
        print(
            f"    → {result.status_label}  ({result.dispatch_count} disp, "
            f"{result.wall_seconds:.1f}s)",
            file=sys.stderr,
        )

    # Baseline diff is only meaningful for full runs.
    deltas = None
    if not suite and not case:
        baseline = load_baseline(baseline_path)
        deltas = diff_against_baseline(results, baseline)

    render_scorecard(results, deltas=deltas, style=style)

    if bless:
        save_baseline(results, baseline_path)
        print(f"\nBaseline written to {baseline_path}", file=sys.stderr)

    return 0 if all(r.passed for r in results) else 1
