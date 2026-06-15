"""Tests for the trace viewer.

The viewer is pure formatting — we verify it loads synthetic JSONL,
groups events into turns correctly, includes the key sections, and
respects --no-color and --summary-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ancestors.trace_viewer import (
    Style,
    Turn,
    group_by_turns,
    latest_trace,
    render,
    resolve_trace_path,
)


def _trace_events() -> list[dict]:
    return [
        {"kind": "on_run_start", "session_id": "abc",
         "extra": {"question": "How many people?"}},
        {"kind": "on_plan", "session_id": "abc",
         "extra": {"turn_index": 1, "rationale": "Start by scoping the corpus.",
                   "calls": [{"tool": "all_individuals", "args": {}, "justification": ""}]}},
        {"kind": "before_dispatch", "session_id": "abc",
         "tool": "all_individuals", "args": {}},
        {"kind": "after_dispatch", "session_id": "abc",
         "tool": "all_individuals", "args": {},
         "result_summary": {"set_handle": "h_1", "set_size": 336},
         "duration_ms": 0.5, "timestamp": 100.0},
        {"kind": "on_assessment", "session_id": "abc",
         "extra": {"turn_index": 1, "decision": "answer",
                   "reasoning": "Read the count from the result.",
                   "facts_added": [{"claim": "There are 336 individuals.",
                                     "confidence": "confirmed"}],
                   "hypotheses_added": [], "dead_ends_added": [],
                   "answer_text": "336 individuals.",
                   "answer_confidence": "confirmed",
                   "stuck_reason": None, "results_seen": 1}},
        {"kind": "on_run_complete", "session_id": "abc",
         "extra": {"question": "How many people?",
                   "terminal_state": "ANSWER",
                   "answer_text": "336 individuals.",
                   "answer_confidence": "confirmed",
                   "stuck_reason": None,
                   "turns": 1, "total_dispatches": 1,
                   "confirmed_facts": [{"claim": "There are 336 individuals.",
                                         "confidence": "confirmed"}],
                   "working_hypotheses": [], "dead_ends": []}},
    ]


def test_group_by_turns_pairs_plan_dispatch_assessment():
    turns = group_by_turns(_trace_events())
    assert len(turns) == 1
    t = turns[0]
    assert t.plan is not None
    assert t.assessment is not None
    assert len(t.dispatches) == 1
    before, after = t.dispatches[0]
    assert after["result_summary"]["set_handle"] == "h_1"


def test_render_contains_question_and_answer():
    out = render(_trace_events(), style=Style(enabled=False), width=80)
    assert "How many people?" in out
    assert "336 individuals." in out
    assert "ANSWER" in out
    assert "confirmed" in out
    assert "Tools used:" in out
    assert "all_individuals×1" in out


def test_render_summary_only_omits_turn_blocks():
    out = render(
        _trace_events(),
        style=Style(enabled=False),
        width=80,
        summary_only=True,
    )
    # Turn rule should not appear.
    assert "Turn 1" not in out
    # But outcome and summary should.
    assert "ANSWER" in out
    assert "Tools used:" in out


def test_render_no_color_emits_no_ansi_escapes():
    out = render(_trace_events(), style=Style(enabled=False), width=80)
    assert "\033[" not in out


def test_render_color_emits_ansi_escapes():
    out = render(_trace_events(), style=Style(enabled=True), width=80)
    assert "\033[" in out


def test_resolve_trace_path_finds_file(tmp_path: Path):
    p = tmp_path / "abcd00112233.jsonl"
    p.write_text("{}\n")
    resolved = resolve_trace_path("abcd00112233", tmp_path)
    assert resolved == p


def test_resolve_trace_path_accepts_filename(tmp_path: Path):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n")
    resolved = resolve_trace_path(str(p), tmp_path)
    assert resolved == p


def test_latest_trace_returns_most_recent(tmp_path: Path):
    older = tmp_path / "aaaaaaaaaaaa.jsonl"
    older.write_text("{}\n")
    # Sleep would be nice but stat mtime granularity is fine on macOS:
    import os
    import time
    os.utime(older, (time.time() - 100, time.time() - 100))
    newer = tmp_path / "bbbbbbbbbbbb.jsonl"
    newer.write_text("{}\n")
    assert latest_trace(tmp_path) == newer


def test_render_handles_stuck_terminal():
    events = list(_trace_events())
    events[-1] = {
        **events[-1],
        "extra": {
            **events[-1]["extra"],
            "terminal_state": "STUCK",
            "answer_text": None,
            "stuck_reason": "no path forward",
        },
    }
    out = render(events, style=Style(enabled=False), width=80)
    assert "STUCK" in out
    assert "no path forward" in out
