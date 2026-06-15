"""CLI entry for the eval harness against the genealogy corpus.

    uv run python -m ancestors.eval                      # run all cases
    uv run python -m ancestors.eval --suite lineage      # one suite
    uv run python -m ancestors.eval --case david_johnston
    uv run python -m ancestors.eval --bless              # overwrite baseline
    uv run python -m ancestors.eval --no-cache           # bypass LLM cache
    uv run python -m ancestors.eval --no-color

This file is the domain-specific entry: it knows about the GEDCOM file
and constructs a GedcomCorpus. The harness it calls is corpus-agnostic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ancestors.corpus import GedcomCorpus
from ancestors.eval.baseline import DEFAULT_BASELINE_PATH
from ancestors.eval.harness import run_eval

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

    print(f"Loading corpus from {args.gedcom}…", file=sys.stderr)
    corpus = GedcomCorpus.from_path(args.gedcom)

    return run_eval(
        corpus,
        suite=args.suite,
        case=args.case,
        no_cache=args.no_cache,
        no_color=args.no_color,
        bless=args.bless,
        baseline_path=args.baseline,
    )


if __name__ == "__main__":
    sys.exit(main())
