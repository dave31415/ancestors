"""Tests for the safe dispatch boundary.

These tests focus on the *guarantees* the dispatcher must enforce:
unknown tools rejected, schema violations rejected, oversize results
rejected, fabricated handles rejected, exceptions converted to structured
errors. Functional correctness of individual tools is tested elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.dispatch import TOOLS, Dispatcher
from ancestors.session import bind_session, clear_session
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"


@pytest.fixture(scope="module", autouse=True)
def session():
    bind_session(load_gedcom(GEDCOM_PATH))
    yield
    clear_session()


@pytest.fixture
def disp():
    return Dispatcher()


def test_unknown_tool_returns_structured_error(disp):
    result = disp.dispatch("rm_rf_root", {})
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "unknown_tool"


def test_invalid_arguments_rejected(disp):
    # Missing required field.
    result = disp.dispatch("get_ancestors_of", {})
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_malformed_person_id_rejected(disp):
    # An attempt to inject a path-like string fails the regex constraint.
    result = disp.dispatch(
        "get_ancestors_of",
        {"person_id": "../../etc/passwd", "max_generations": 1},
    )
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_extra_arguments_rejected(disp):
    # extra='forbid' on each args model — typo'd or smuggled fields fail.
    result = disp.dispatch(
        "get_ancestors_of",
        {"person_id": DAVID_ID, "max_generations": 1, "shell_command": "rm -rf /"},
    )
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_excessive_generations_rejected(disp):
    result = disp.dispatch(
        "get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 9999}
    )
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_excessive_hydration_limit_rejected(disp):
    a = disp.dispatch("all_individuals", {})
    result = disp.dispatch(
        "get_individuals", {"ids": a.set_handle, "limit": 100_000}
    )
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_unknown_set_handle_rejected(disp):
    result = disp.dispatch("count", {"ids": "h_9999999"})
    assert result.ok is False
    assert result.error.code == "unknown_set_handle"


def test_fabricated_handle_pattern_rejected(disp):
    # The agent cannot smuggle a raw id list in via a SetHandle field.
    result = disp.dispatch("count", {"ids": "h_../etc/passwd"})
    assert result.ok is False
    assert result.error.code == "invalid_arguments"


def test_successful_pipeline_returns_handles(disp):
    a = disp.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 5})
    assert a.ok and a.set_handle is not None and a.set_size > 1

    b = disp.dispatch(
        "filter_by_event_place",
        {"ids": a.set_handle, "event_type": "BIRT", "place_contains": "Canada"},
    )
    assert b.ok and b.set_handle is not None
    assert b.set_size <= a.set_size

    c = disp.dispatch("count", {"ids": b.set_handle})
    assert c.ok and c.value == b.set_size


def test_call_budget_enforced():
    d = Dispatcher()
    from ancestors.dispatch import MAX_CALLS_PER_SESSION

    d.call_count = MAX_CALLS_PER_SESSION
    r = d.dispatch("all_individuals", {})
    assert r.ok is False
    assert r.error.code == "call_budget_exceeded"


def test_memoization_returns_same_handle_for_repeat_call(disp):
    a = disp.dispatch("all_individuals", {})
    b = disp.dispatch("all_individuals", {})
    assert a.set_handle == b.set_handle
    assert a.set_size == b.set_size


def test_memoization_caches_errors_too(disp):
    a = disp.dispatch("does_not_exist", {})
    b = disp.dispatch("does_not_exist", {})
    assert a.ok is False and b.ok is False
    assert a.error.code == b.error.code


def test_memoization_distinguishes_different_args(disp):
    a = disp.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 3})
    b = disp.dispatch("get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 5})
    assert a.set_handle != b.set_handle


def test_handles_summary_lists_each_handle(disp):
    a = disp.dispatch("all_individuals", {})
    b = disp.dispatch(
        "filter_by_surname", {"ids": a.set_handle, "surname": "Johnston"}
    )
    summary = disp.handles_summary()
    assert a.set_handle in summary
    assert b.set_handle in summary
    assert "all_individuals" in summary
    assert "filter_by_surname" in summary


def test_handles_summary_empty_when_no_handles_produced(disp):
    assert "no IdSet handles" in disp.handles_summary()


def test_validation_error_names_the_field(disp):
    # filter_by_event_place requires event_type — its omission should name it.
    a = disp.dispatch("all_individuals", {})
    result = disp.dispatch(
        "filter_by_event_place",
        {"ids": a.set_handle, "place_contains": "Ireland"},
    )
    assert result.ok is False
    assert result.error.code == "invalid_arguments"
    msg = result.error.message
    assert "event_type" in msg
    # Allowed values should be hinted somewhere in the message.
    assert "BIRT" in msg


def test_validation_error_names_unknown_enum_value(disp):
    a = disp.dispatch("all_individuals", {})
    result = disp.dispatch(
        "filter_by_event_place",
        {"ids": a.set_handle, "event_type": "NOT_A_REAL_EVENT", "place_contains": "x"},
    )
    assert result.ok is False
    msg = result.error.message
    assert "event_type" in msg


def test_tool_registry_has_no_dangerous_callables():
    """Belt-and-braces: ensure no registered tool function is exec/eval/open."""
    dangerous = {"exec", "eval", "compile", "open", "system", "spawn"}
    for tool in TOOLS.values():
        assert tool.fn.__name__ not in dangerous
        # And the function must live under our package, not stdlib.
        assert tool.fn.__module__.startswith("ancestors.")
