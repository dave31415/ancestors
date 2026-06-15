"""CLI entry for the eval harness.

    uv run python -m ancestors.eval                      # run all cases
    uv run python -m ancestors.eval --suite lineage      # one suite
    uv run python -m ancestors.eval --case david_johnston
    uv run python -m ancestors.eval --bless              # overwrite baseline
    uv run python -m ancestors.eval --no-cache           # bypass LLM cache
    uv run python -m ancestors.eval --no-color
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ancestors.eval.baseline import (
    DEFAULT_BASELINE_PATH,
    diff_against_baseline,
    load_baseline,
    save_baseline,
)
from ancestors.eval.cases import all_cases, case_by_name, cases_in_suite
from ancestors.eval.reporter import render_scorecard
from ancestors.eval.runner import load_corpus, run_case
from ancestors.llm import CachedAnthropic
from ancestors.trace_viewer import Style

DEFAULT_GEDCOM_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "export-Ancestors.ged"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ancestors.eval",
        description="Run the genealogy agent eval harness.",
    )
    p.add_argument("--suite", help="Only run cases from this suite.")
    p.add_argument(
        "--case",
        help="Only run one case (by short name or suite/name).",
    )
    p.add_argument(
        "--gedcom",
        type=Path,
        default=DEFAULT_GEDCOM_PATH,
        help="Path to the GEDCOM file (default: data/export-Ancestors.ged).",
    )
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the disk LLM cache.")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour. Auto-off when stdout isn't a TTY.")
    p.add_argument("--bless", action="store_true",
                   help="Overwrite baseline.json with this run's results.")
    p.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Where the baseline JSON lives.",
    )
    args = p.parse_args(argv)

    style = Style(enabled=not args.no_color and sys.stdout.isatty())

    # Select cases.
    if args.case:
        c = case_by_name(args.case)
        if c is None:
            print(f"error: case {args.case!r} not found", file=sys.stderr)
            return 1
        cases = [c]
    elif args.suite:
        cases = cases_in_suite(args.suite)
        if not cases:
            print(f"error: no cases in suite {args.suite!r}", file=sys.stderr)
            return 1
    else:
        cases = all_cases()

    print(f"Loading corpus from {args.gedcom}…", file=sys.stderr)
    db = load_corpus(args.gedcom)
    llm = CachedAnthropic(use_cache=not args.no_cache)

    results = []
    for i, case in enumerate(cases, 1):
        print(
            f"[{i}/{len(cases)}] {case.fully_qualified_name}…",
            file=sys.stderr,
            flush=True,
        )
        result = run_case(case, db, llm)
        results.append(result)
        # Newline so the [N/M] line isn't overwritten by subsequent output.
        print(f"    → {result.status_label}  ({result.dispatch_count} disp, "
              f"{result.wall_seconds:.1f}s)", file=sys.stderr)

    # Baseline diff (only meaningful for full runs).
    deltas = None
    if not args.suite and not args.case:
        baseline = load_baseline(args.baseline)
        deltas = diff_against_baseline(results, baseline)

    render_scorecard(results, deltas=deltas, style=style)

    if args.bless:
        save_baseline(results, args.baseline)
        print(f"\nBaseline written to {args.baseline}", file=sys.stderr)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
