"""Tests for the eval check primitives and baseline diff logic.

We don't run the loop here — we synthesise LoopResult objects and verify
each check passes or fails the way we expect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ancestors.agent.loop import LoopResult, StateName
from ancestors.agent.state import ConfidenceLevel, new_state
from ancestors.eval.baseline import (
    diff_against_baseline,
    load_baseline,
    save_baseline,
    to_baseline_entry,
)
from ancestors.eval.case import Case, CaseResult
from ancestors.eval.checks import (
    answer_does_not_mention,
    answer_mentions,
    answer_mentions_all,
    answer_mentions_any,
    answer_mentions_year,
    confidence_at_least,
    confidence_at_most,
    dispatches_at_most,
    terminates_at,
    terminates_at_one_of,
)


def _result(
    *,
    state: str = "ANSWER",
    answer: str | None = "ok",
    confidence: ConfidenceLevel | None = ConfidenceLevel.CONFIRMED,
    dispatches: int = 1,
    turns: int = 1,
) -> LoopResult:
    return LoopResult(
        terminal_state=StateName(state),
        answer_text=answer,
        answer_confidence=confidence,
        stuck_reason=None,
        final_state=new_state("?"),
        turns=turns,
        total_dispatches=dispatches,
    )


# ---------------------------------------------------------------------------
# terminates_at
# ---------------------------------------------------------------------------


def test_terminates_at_passes_when_matching():
    c = terminates_at("ANSWER")
    cr = c(_result(state="ANSWER"), [])
    assert cr.passed


def test_terminates_at_fails_when_not_matching():
    c = terminates_at("ANSWER")
    cr = c(_result(state="STUCK"), [])
    assert not cr.passed


def test_terminates_at_one_of():
    c = terminates_at_one_of(["ANSWER", "STUCK"])
    assert c(_result(state="ANSWER"), []).passed
    assert c(_result(state="STUCK"), []).passed


# ---------------------------------------------------------------------------
# answer text
# ---------------------------------------------------------------------------


def test_answer_mentions_is_case_insensitive():
    c = answer_mentions("Johnston")
    assert c(_result(answer="david edmund johnston is the answer"), []).passed


def test_answer_mentions_year_finds_year():
    c = answer_mentions_year(1975)
    assert c(_result(answer="born 1975 in MA"), []).passed
    assert not c(_result(answer="no year here"), []).passed


def test_answer_mentions_all():
    c = answer_mentions_all(["David", "Johnston"])
    assert c(_result(answer="David Johnston is the answer"), []).passed
    cr = c(_result(answer="David alone"), [])
    assert not cr.passed
    assert "Johnston" in cr.message


def test_answer_mentions_any():
    c = answer_mentions_any(["Trek", "Specialized"])
    assert c(_result(answer="The Trek bike is best"), []).passed
    assert not c(_result(answer="genealogy"), []).passed


def test_answer_does_not_mention():
    c = answer_does_not_mention("Trek")
    assert c(_result(answer="genealogy answer"), []).passed
    assert not c(_result(answer="Trek bikes"), []).passed


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------


def test_confidence_at_least_passes_when_high_enough():
    c = confidence_at_least("probable")
    assert c(_result(confidence=ConfidenceLevel.CONFIRMED), []).passed
    assert c(_result(confidence=ConfidenceLevel.PROBABLE), []).passed
    assert not c(_result(confidence=ConfidenceLevel.POSSIBLE), []).passed


def test_confidence_at_least_fails_when_unset():
    c = confidence_at_least("probable")
    cr = c(_result(confidence=None), [])
    assert not cr.passed


def test_confidence_at_most_passes_when_capped():
    c = confidence_at_most("possible")
    assert c(_result(confidence=ConfidenceLevel.SPECULATIVE), []).passed
    assert c(_result(confidence=ConfidenceLevel.POSSIBLE), []).passed
    assert not c(_result(confidence=ConfidenceLevel.PROBABLE), []).passed


def test_confidence_at_most_passes_when_none_emitted():
    # E.g. STUCK terminal — confidence is None.
    c = confidence_at_most("possible")
    assert c(_result(confidence=None), []).passed


# ---------------------------------------------------------------------------
# dispatches
# ---------------------------------------------------------------------------


def test_dispatches_at_most():
    c = dispatches_at_most(5)
    assert c(_result(dispatches=5), []).passed
    assert not c(_result(dispatches=6), []).passed


# ---------------------------------------------------------------------------
# baseline diff
# ---------------------------------------------------------------------------


def _case_result(name: str, suite: str, status: str, dispatches: int) -> CaseResult:
    case = Case(name=name, suite=suite, question="?", checks=[])
    state = StateName.STUCK if status == "STUCK" else StateName.ANSWER
    loop_result = LoopResult(
        terminal_state=state,
        answer_text="x" if status != "STUCK" else None,
        answer_confidence=ConfidenceLevel.CONFIRMED if status == "PASS" else None,
        stuck_reason="?" if status == "STUCK" else None,
        final_state=new_state("?"),
        turns=1,
        total_dispatches=dispatches,
    )
    # Synthesise check results matching the status.
    from ancestors.eval.case import CheckResult

    checks = (
        [CheckResult(True, "ok", "x")]
        if status == "PASS"
        else [CheckResult(False, "fail", "x")] if status == "FAIL"
        else []
    )
    return CaseResult(
        case=case,
        loop_result=loop_result,
        check_results=checks,
        wall_seconds=0.0,
        dispatch_count=dispatches,
        llm_call_count=2,
        tools_used=[],
    )


def test_diff_against_baseline_detects_status_change():
    current = [_case_result("a", "lookup", "FAIL", 5)]
    baseline = {
        "version": 1,
        "cases": {
            "lookup/a": {"status": "PASS", "dispatches": 5},
        },
    }
    deltas = diff_against_baseline(current, baseline)
    kinds = [d.kind for d in deltas]
    assert "status_change" in kinds


def test_diff_against_baseline_detects_efficiency_drift():
    current = [_case_result("a", "lookup", "PASS", 12)]
    baseline = {
        "version": 1,
        "cases": {
            "lookup/a": {"status": "PASS", "dispatches": 5},
        },
    }
    deltas = diff_against_baseline(current, baseline)
    assert any(d.kind == "efficiency_change" for d in deltas)


def test_diff_against_baseline_detects_new_cases():
    current = [_case_result("a", "lookup", "PASS", 5)]
    baseline = {"version": 1, "cases": {}}
    deltas = diff_against_baseline(current, baseline)
    assert any(d.kind == "new" for d in deltas)


def test_diff_against_baseline_detects_removed():
    current = [_case_result("a", "lookup", "PASS", 5)]
    baseline = {
        "version": 1,
        "cases": {
            "lookup/a": {"status": "PASS", "dispatches": 5},
            "lookup/b_removed": {"status": "PASS", "dispatches": 3},
        },
    }
    deltas = diff_against_baseline(current, baseline)
    assert any(d.kind == "removed" for d in deltas)


def test_save_and_load_baseline_roundtrip(tmp_path: Path):
    current = [_case_result("a", "lookup", "PASS", 5)]
    path = tmp_path / "baseline.json"
    save_baseline(current, path)
    loaded = load_baseline(path)
    assert loaded["cases"]["lookup/a"]["status"] == "PASS"
    assert loaded["cases"]["lookup/a"]["dispatches"] == 5


def test_load_baseline_returns_none_when_missing(tmp_path: Path):
    assert load_baseline(tmp_path / "missing.json") is None
