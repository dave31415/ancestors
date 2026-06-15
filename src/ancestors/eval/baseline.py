"""Baseline tracking — what makes the eval harness compound.

The baseline records, per case: status (PASS/FAIL/STUCK), dispatch count,
LLM call count, and confidence. Subsequent runs compare against it and
emit a diff: improvements, regressions, efficiency drift.

The `bless` mode overwrites the baseline with the current run — used after
intentional improvements to capture the new floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ancestors.eval.case import CaseResult

DEFAULT_BASELINE_PATH = (
    Path(__file__).parent / "baselines" / "baseline.json"
)


@dataclass
class CaseDelta:
    """How a case changed compared to baseline."""

    case_name: str
    kind: str  # "new", "removed", "status_change", "efficiency_change", "unchanged"
    detail: str


def to_baseline_entry(result: CaseResult) -> dict[str, Any]:
    """Serialise one CaseResult to the baseline schema."""
    confidence = result.loop_result.answer_confidence
    return {
        "status": result.status_label,
        "dispatches": result.dispatch_count,
        "llm_calls": result.llm_call_count,
        "confidence": confidence.value if confidence is not None else None,
        "tools_used": result.tools_used,
    }


def save_baseline(results: list[CaseResult], path: Path = DEFAULT_BASELINE_PATH) -> None:
    """Write the current run as the blessed baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": {
            r.case.fully_qualified_name: to_baseline_entry(r)
            for r in results
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> dict[str, Any] | None:
    """Read the baseline, or None if absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def diff_against_baseline(
    results: list[CaseResult],
    baseline: dict[str, Any] | None,
) -> list[CaseDelta]:
    """Compare a run's results to the baseline and report deltas.

    Categorises each case as new / removed / status_change / efficiency_change
    / unchanged. The reporter renders these grouped.
    """
    if baseline is None:
        return [
            CaseDelta(r.case.fully_qualified_name, "new", "no prior baseline")
            for r in results
        ]

    baseline_cases = baseline.get("cases", {})
    current_names = {r.case.fully_qualified_name for r in results}
    baseline_names = set(baseline_cases.keys())

    deltas: list[CaseDelta] = []

    # Cases removed from the suite since baseline.
    for name in baseline_names - current_names:
        deltas.append(CaseDelta(name, "removed", "in baseline but not in current run"))

    # Cases added since baseline.
    for r in results:
        name = r.case.fully_qualified_name
        if name not in baseline_names:
            deltas.append(CaseDelta(name, "new", f"new case ({r.status_label})"))
            continue
        deltas.extend(_compare_one(r, baseline_cases[name]))

    return deltas


def _compare_one(result: CaseResult, baseline_entry: dict[str, Any]) -> list[CaseDelta]:
    name = result.case.fully_qualified_name
    deltas: list[CaseDelta] = []

    current_status = result.status_label
    baseline_status = baseline_entry.get("status")
    if current_status != baseline_status:
        deltas.append(
            CaseDelta(
                name,
                "status_change",
                f"{baseline_status} → {current_status}",
            )
        )

    current_dispatch = result.dispatch_count
    baseline_dispatch = baseline_entry.get("dispatches")
    if isinstance(baseline_dispatch, int) and baseline_dispatch != current_dispatch:
        delta = current_dispatch - baseline_dispatch
        sign = "+" if delta > 0 else ""
        deltas.append(
            CaseDelta(
                name,
                "efficiency_change",
                f"dispatches: {baseline_dispatch} → {current_dispatch} ({sign}{delta})",
            )
        )

    if not deltas:
        deltas.append(CaseDelta(name, "unchanged", current_status))

    return deltas
