"""Check primitive factories.

Each factory returns a `Check` — a callable that takes (LoopResult,
trace_events) and returns a CheckResult. Factories assign a readable
`name` to the returned check so failure messages identify which
assertion broke.

The vocabulary is intentionally small. Add a new primitive only when
several cases want the same shape; one-off assertions can be inline
functions in the case file.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

from ancestors.agent.loop import LoopResult
from ancestors.agent.observability import TraceEvent
from ancestors.agent.state import ConfidenceLevel
from ancestors.eval.case import Check, CheckResult

# Confidence ordering for at_least / at_most comparisons. `refuted` is
# deliberately omitted — it's not on the same axis as "how confident am I
# the claim is true."
_CONFIDENCE_ORDER = [
    "speculative",
    "possible",
    "probable",
    "confirmed",
]


def _named(name: str):
    """Decorator factory: assign a stable name to a check."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
            inner = fn(result, events)
            return CheckResult(
                passed=inner.passed,
                message=inner.message,
                name=name,
            )

        return wrapper

    return decorate


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------


def terminates_at(state: str) -> Check:
    @_named(f"terminates_at({state!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        got = result.terminal_state.value
        return CheckResult(
            passed=(got == state),
            message=f"terminal_state={got!r}",
        )

    return check


def terminates_at_one_of(states: Iterable[str]) -> Check:
    states = list(states)

    @_named(f"terminates_at_one_of({states!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        got = result.terminal_state.value
        return CheckResult(
            passed=(got in states),
            message=f"terminal_state={got!r}",
        )

    return check


# ---------------------------------------------------------------------------
# Answer text
# ---------------------------------------------------------------------------


def answer_mentions(substring: str, *, case_sensitive: bool = False) -> Check:
    @_named(f"answer_mentions({substring!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        text = result.answer_text or ""
        if not case_sensitive:
            ok = substring.lower() in text.lower()
        else:
            ok = substring in text
        snippet = (text[:80] + "…") if len(text) > 80 else text
        return CheckResult(
            passed=ok,
            message=f"answer={snippet!r}",
        )

    return check


def answer_mentions_all(substrings: Iterable[str]) -> Check:
    """All substrings must appear (case-insensitive)."""
    subs = list(substrings)

    @_named(f"answer_mentions_all({subs!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        text = (result.answer_text or "").lower()
        missing = [s for s in subs if s.lower() not in text]
        return CheckResult(
            passed=(not missing),
            message=f"missing={missing!r}" if missing else "all present",
        )

    return check


def answer_mentions_any(substrings: Iterable[str]) -> Check:
    """At least one substring must appear (case-insensitive)."""
    subs = list(substrings)

    @_named(f"answer_mentions_any({subs!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        text = (result.answer_text or "").lower()
        hits = [s for s in subs if s.lower() in text]
        return CheckResult(
            passed=bool(hits),
            message=f"hits={hits!r}" if hits else "none of the substrings present",
        )

    return check


def answer_does_not_mention(substring: str) -> Check:
    @_named(f"answer_does_not_mention({substring!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        text = (result.answer_text or "").lower()
        return CheckResult(
            passed=(substring.lower() not in text),
            message=f"forbidden substring present" if substring.lower() in text else "absent",
        )

    return check


def answer_mentions_year(year: int) -> Check:
    """Year shows up as a 4-digit number in the answer."""
    return answer_mentions(str(year))


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


def _conf_index(level: str | ConfidenceLevel | None) -> int | None:
    if level is None:
        return None
    value = level.value if isinstance(level, ConfidenceLevel) else level
    try:
        return _CONFIDENCE_ORDER.index(value)
    except ValueError:
        return None  # "refuted" or unknown — off-axis


def confidence_at_least(level: str) -> Check:
    target_idx = _CONFIDENCE_ORDER.index(level)

    @_named(f"confidence_at_least({level!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        got_idx = _conf_index(result.answer_confidence)
        got_str = (
            result.answer_confidence.value
            if result.answer_confidence is not None
            else "None"
        )
        if got_idx is None:
            return CheckResult(
                passed=False,
                message=f"confidence={got_str!r} (off the on-axis vocabulary)",
            )
        return CheckResult(
            passed=(got_idx >= target_idx),
            message=f"confidence={got_str!r}",
        )

    return check


def confidence_at_most(level: str) -> Check:
    """Used for out-of-scope / refusal cases — confidence should be capped."""
    target_idx = _CONFIDENCE_ORDER.index(level)

    @_named(f"confidence_at_most({level!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        if result.answer_confidence is None:
            # No confidence emitted (e.g. STUCK) — treat as bounded.
            return CheckResult(
                passed=True,
                message="no confidence (STUCK or refused)",
            )
        got_idx = _conf_index(result.answer_confidence)
        got_str = result.answer_confidence.value
        if got_idx is None:
            return CheckResult(
                passed=True,
                message=f"confidence={got_str!r} (off-axis, allowed)",
            )
        return CheckResult(
            passed=(got_idx <= target_idx),
            message=f"confidence={got_str!r}",
        )

    return check


# ---------------------------------------------------------------------------
# Cost / efficiency
# ---------------------------------------------------------------------------


def dispatches_at_most(n: int) -> Check:
    @_named(f"dispatches_at_most({n})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        got = result.total_dispatches
        return CheckResult(
            passed=(got <= n),
            message=f"total_dispatches={got}",
        )

    return check


def tools_used_include(tool: str) -> Check:
    @_named(f"tools_used_include({tool!r})")
    def check(result: LoopResult, events: list[TraceEvent]) -> CheckResult:
        used = {
            (e.tool or "")
            for e in events
            if e.kind == "before_dispatch" and e.tool
        }
        return CheckResult(
            passed=(tool in used),
            message=f"tools_used={sorted(used)}",
        )

    return check


# ---------------------------------------------------------------------------
# Convenience compositions
# ---------------------------------------------------------------------------


def custom(fn, *, name: str = "custom") -> Check:
    """Wrap a one-off function as a Check with a stable name."""
    return _named(name)(fn)


__all__ = [
    "terminates_at",
    "terminates_at_one_of",
    "answer_mentions",
    "answer_mentions_all",
    "answer_mentions_any",
    "answer_does_not_mention",
    "answer_mentions_year",
    "confidence_at_least",
    "confidence_at_most",
    "dispatches_at_most",
    "tools_used_include",
    "custom",
]
