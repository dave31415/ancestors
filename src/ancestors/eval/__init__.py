"""Evaluation harness for the genealogy agent.

Deterministic, no LLM-judge. Composable check primitives. Disk-cached
LLM client makes second runs of unchanged cases near-instant.

Layout:
    case.py       — Case / CheckResult / CaseResult dataclasses
    checks.py     — composable check primitive factories
    runner.py     — run_case, run_suite
    reporter.py   — render scorecard to terminal
    baseline.py   — load/save/diff baseline.json
    cases/        — auto-discovered question modules (lookup/lineage/…)
    baselines/    — committed baseline scores per case

CLI: python -m ancestors.eval
"""
