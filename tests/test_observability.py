"""Tests for the observability hooks.

We attach an in-memory collector to a Dispatcher, run a small pipeline,
and verify the correct events fire in the correct order with the
expected fields populated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ancestors.agent.observability import (
    HookRegistry,
    JsonlTraceWriter,
    make_collector,
)
from ancestors.dispatch import Dispatcher
from ancestors.session import bind_session, clear_session
from ancestors.tool_registry import TOOLS
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"


@pytest.fixture(scope="module", autouse=True)
def session():
    bind_session(load_gedcom(GEDCOM_PATH))
    yield
    clear_session()


@pytest.fixture
def dispatcher_with_collector():
    events, observer = make_collector()
    hooks = HookRegistry()
    hooks.subscribe(observer)
    return Dispatcher(tools=TOOLS, hooks=hooks), events


def test_successful_call_emits_before_and_after(dispatcher_with_collector):
    d, events = dispatcher_with_collector
    d.dispatch("all_individuals", {})
    assert [e.kind for e in events] == ["before_dispatch", "after_dispatch"]
    after = events[1]
    assert after.tool == "all_individuals"
    assert after.duration_ms is not None
    assert after.result_summary["set_size"] == 336


def test_failed_call_emits_before_and_on_error(dispatcher_with_collector):
    d, events = dispatcher_with_collector
    d.dispatch("does_not_exist", {})
    assert [e.kind for e in events] == ["before_dispatch", "on_error"]
    err = events[1]
    assert err.error_code == "unknown_tool"


def test_observer_exception_does_not_break_dispatch(dispatcher_with_collector):
    d, events = dispatcher_with_collector

    def broken_observer(event):
        raise RuntimeError("broken")

    d.hooks.subscribe(broken_observer)
    result = d.dispatch("all_individuals", {})
    assert result.ok is True


def test_no_hooks_still_dispatches():
    d = Dispatcher(tools=TOOLS)  # no hooks attached
    result = d.dispatch("all_individuals", {})
    assert result.ok is True


def test_jsonl_writer_creates_file(tmp_path: Path):
    writer = JsonlTraceWriter(trace_dir=tmp_path)
    hooks = HookRegistry()
    hooks.subscribe(writer)
    d = Dispatcher(tools=TOOLS, hooks=hooks)
    d.dispatch("all_individuals", {})
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "before_dispatch" in content
    assert "after_dispatch" in content
