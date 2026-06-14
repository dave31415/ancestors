"""Hooks and observers for the dispatcher and (later) the agent loop.

Built upfront, not bolted on. menu_agent's monkey-patched logging is the
cautionary tale: when the observer interface doesn't exist, the first thing
that needs to listen to events tries to hot-swap a private method, and from
that point on every observer is some variant of monkey-patch.

The design here is deliberately minimal:
- Events are dataclasses with a discriminant.
- An Observer is anything that consumes events (a callable or a class with
  __call__).
- The Dispatcher takes a list of observers at construction time and emits
  events at well-defined lifecycle points.

The default `JsonlTraceWriter` writes one JSON line per event to a file —
enough to drive evaluation and post-hoc analysis without dragging in a
logging framework.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

log = logging.getLogger(__name__)

EventKind = Literal[
    # Dispatcher lifecycle.
    "before_dispatch",
    "after_dispatch",
    "on_error",
    # Loop lifecycle. The loop emits these so traces are self-contained —
    # the question, the model's plan rationale, the model's assessment, and
    # the final answer or stuck reason all show up in the JSONL trace
    # without needing the demo's stdout.
    "on_run_start",
    "on_state_transition",
    "on_plan",
    "on_assessment",
    "on_run_complete",
]


@dataclass
class TraceEvent:
    """A single observable event from the dispatcher or agent loop."""

    kind: EventKind
    timestamp: float
    session_id: str
    # Dispatcher events
    tool: str | None = None
    args: dict[str, Any] | None = None
    result_summary: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    # Loop events
    from_state: str | None = None
    to_state: str | None = None
    turn_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Observer(Protocol):
    """A consumer of TraceEvents. Plain callables satisfy this."""

    def __call__(self, event: TraceEvent) -> None: ...


@dataclass
class HookRegistry:
    """Owns observers and emits events to them.

    Per-instance, not a process-wide singleton — a session owns its hooks.
    Observer failures are caught and logged; one bad observer can't break
    the agent.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    observers: list[Observer] = field(default_factory=list)

    def subscribe(self, observer: Observer) -> None:
        self.observers.append(observer)

    def emit(self, kind: EventKind, **fields: Any) -> None:
        event = TraceEvent(
            kind=kind,
            timestamp=time.time(),
            session_id=self.session_id,
            **fields,
        )
        for obs in self.observers:
            try:
                obs(event)
            except Exception:  # noqa: BLE001 — an observer mustn't kill the agent
                log.exception("Observer %r raised on %s", obs, kind)


# ---------------------------------------------------------------------------
# Built-in observers
# ---------------------------------------------------------------------------


@dataclass
class JsonlTraceWriter:
    """Append one JSON line per event to a file.

    Default trace location: ./traces/<session_id>.jsonl. Created on first event.
    The file is opened in append mode every emit so a crash mid-session still
    leaves a partial-but-valid trace on disk.
    """

    trace_dir: Path = field(default_factory=lambda: Path("traces"))
    path: Path | None = None

    def __call__(self, event: TraceEvent) -> None:
        if self.path is None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.trace_dir / f"{event.session_id}.jsonl"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), default=str) + os.linesep)


def make_collector() -> tuple[list[TraceEvent], Observer]:
    """In-memory event collector. Useful in tests."""
    collected: list[TraceEvent] = []

    def observer(event: TraceEvent) -> None:
        collected.append(event)

    return collected, observer
