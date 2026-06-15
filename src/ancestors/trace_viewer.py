"""Pretty-print a JSONL session trace to the terminal.

Pure stdlib. No API calls. Reads a trace produced by JsonlTraceWriter,
groups events by turn, and renders the question, each turn's plan +
dispatches + assessment, and the outcome. Designed to be the first thing
a researcher reaches for when inspecting a past session.

Usage:
    python -m ancestors.trace_viewer                 # latest trace in ./traces
    python -m ancestors.trace_viewer SESSION_OR_FILE
    python -m ancestors.trace_viewer --summary-only
    python -m ancestors.trace_viewer --no-color

The viewer is intentionally rendered from the captured text — the model's
own plan rationales and assessment reasoning surface verbatim, not via
re-summarisation. What you read here is what the agent emitted at the
moment it emitted it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TRACES_DIR = Path("traces")
SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# ANSI colour helpers — no-op when colour is disabled.
# ---------------------------------------------------------------------------


@dataclass
class Style:
    enabled: bool = True

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, t: str) -> str: return self._wrap("1", t)
    def dim(self, t: str) -> str: return self._wrap("2", t)
    def red(self, t: str) -> str: return self._wrap("31", t)
    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)

    def confidence(self, value: str | None) -> str:
        if value is None:
            return self.dim("(unset)")
        colour_map = {
            "confirmed": self.green,
            "probable": self.green,
            "possible": self.yellow,
            "speculative": self.yellow,
            "refuted": self.red,
        }
        fn = colour_map.get(value, self.dim)
        return fn(value)


# ---------------------------------------------------------------------------
# Loading & grouping
# ---------------------------------------------------------------------------


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL trace into a list of event dicts."""
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


@dataclass
class Turn:
    """One PLAN -> EXEC* -> ASSESS cycle, as observed in the trace."""

    index: int
    plan: dict[str, Any] | None = None
    dispatches: list[tuple[dict[str, Any], dict[str, Any] | None]] = field(
        default_factory=list
    )
    assessment: dict[str, Any] | None = None


def group_by_turns(events: list[dict[str, Any]]) -> list[Turn]:
    """Walk events in order, attaching each to the current turn.

    A turn is bracketed by `on_plan` (start) and `on_assessment` (end).
    Dispatches that fire between are attributed to it. before_dispatch and
    after_dispatch events are paired by tool+args order.
    """
    turns: list[Turn] = []
    current: Turn | None = None
    pending_before: dict[str, Any] | None = None

    for event in events:
        kind = event.get("kind")
        if kind == "on_plan":
            current = Turn(index=event["extra"].get("turn_index", len(turns) + 1))
            current.plan = event
            turns.append(current)
        elif kind == "on_assessment" and current is not None:
            current.assessment = event
        elif kind == "before_dispatch":
            pending_before = event
        elif kind == "after_dispatch" or kind == "on_error":
            if current is not None:
                current.dispatches.append((pending_before or event, event))
            pending_before = None

    return turns


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(
    events: list[dict[str, Any]],
    *,
    style: Style,
    width: int = 80,
    summary_only: bool = False,
) -> str:
    """Render the trace as a single multi-line string."""
    rule_double = "═" * width
    rule_single = "─" * width

    lines: list[str] = []

    # Header + question
    session_id = events[0].get("session_id", "?") if events else "?"
    run_start = _first_event(events, "on_plan_start") or _first_event(events, "on_run_start")
    question = ""
    if run_start is not None:
        question = (run_start.get("extra") or {}).get("question", "")
    lines.append(style.bold(rule_double))
    lines.append(style.bold(f"Session  {session_id}"))
    lines.append(style.bold(rule_double))
    lines.append("")
    if question:
        wrapped = textwrap.wrap(question, width=width - 3)
        if wrapped:
            lines.append(f"Q. {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"   {cont}")
    lines.append("")

    # Turn blocks
    turns = group_by_turns(events)
    if not summary_only:
        for turn in turns:
            lines.extend(_render_turn(turn, style=style, width=width))
            lines.append("")

    # Outcome
    complete = _first_event(events, "on_run_complete")
    if complete is not None:
        lines.extend(_render_outcome(complete, style=style, width=width))

    # Summary
    lines.append("")
    lines.append(style.bold(rule_double))
    lines.append(style.bold("Summary"))
    lines.append(style.bold(rule_double))
    lines.extend(_render_summary(events, turns, style=style, width=width))

    return "\n".join(lines)


def _render_turn(turn: Turn, *, style: Style, width: int) -> list[str]:
    lines: list[str] = []
    header = f"─── Turn {turn.index} {'─' * (width - len(f'─── Turn {turn.index} '))}"
    lines.append(style.dim(header))

    # PLAN
    if turn.plan is not None:
        plan_extra = turn.plan.get("extra") or {}
        calls = plan_extra.get("calls") or []
        rationale = plan_extra.get("rationale") or ""

        if calls:
            for i, c in enumerate(calls):
                prefix = style.bold("PLAN") + "  " if i == 0 else "      "
                call_str = _format_call(c.get("tool", "?"), c.get("args") or {}, style=style)
                lines.append(f"{prefix}→ {call_str}")
        else:
            lines.append(style.bold("PLAN") + "  (no tool calls emitted)")
        if rationale:
            wrapped = textwrap.wrap(
                rationale,
                width=width - 18,
                initial_indent="rationale: ",
                subsequent_indent="           ",
            )
            for w in wrapped:
                lines.append(f"        {w}")
        lines.append("")

    # EXEC
    if turn.dispatches:
        for i, (before, after) in enumerate(turn.dispatches):
            tool = (after or before).get("tool", "?")
            args = (after or before).get("args") or {}
            call_str = _format_call(tool, args, style=style)
            prefix = style.bold("EXEC") + "  " if i == 0 else "      "
            dur = (after or {}).get("duration_ms")
            dur_str = f"({dur:.0f} ms)" if dur is not None else ""
            if after is not None and after.get("kind") == "on_error":
                code = after.get("error_code")
                msg = after.get("error_message") or ""
                result = style.red(f"ERROR {code}") + (f": {msg}" if msg else "")
                lines.append(f"{prefix}{call_str}")
                lines.append(f"        {result}  {style.dim(dur_str)}")
            else:
                rs = (after or {}).get("result_summary") or {}
                result = _format_result_summary(rs, style=style)
                lines.append(f"{prefix}{call_str}  →  {result}  {style.dim(dur_str)}")
        lines.append("")

    # ASSESS
    if turn.assessment is not None:
        ax = turn.assessment.get("extra") or {}
        decision = ax.get("decision", "?")
        facts = ax.get("facts_added") or []
        hyps = ax.get("hypotheses_added") or []
        deads = ax.get("dead_ends_added") or []
        reasoning = ax.get("reasoning") or ""

        deltas = []
        if facts: deltas.append(f"facts+={len(facts)}")
        if hyps: deltas.append(f"hyp+={len(hyps)}")
        if deads: deltas.append(f"dead+={len(deads)}")
        deltas_str = "  " + " ".join(deltas) if deltas else ""

        decision_str = _decision_colour(decision, style)
        lines.append(style.bold("ASSESS") + f"  {decision_str}{deltas_str}")
        if reasoning:
            wrapped = textwrap.wrap(reasoning, width=width - 10)
            for w in wrapped:
                lines.append(f"        {w}")

    return lines


def _render_outcome(
    complete: dict[str, Any], *, style: Style, width: int
) -> list[str]:
    rule = "═" * width
    x = complete.get("extra") or {}
    terminal = x.get("terminal_state", "?")
    answer_text = x.get("answer_text")
    confidence = x.get("answer_confidence")
    stuck_reason = x.get("stuck_reason")
    facts = x.get("confirmed_facts") or []
    hyps = x.get("working_hypotheses") or []
    dead_ends = x.get("dead_ends") or []

    out: list[str] = []
    out.append(style.bold(rule))
    if terminal == "ANSWER":
        out.append(
            style.bold("Outcome  ")
            + style.green("ANSWER")
            + f"  (confidence: {style.confidence(confidence)})"
        )
    elif terminal == "STUCK":
        out.append(style.bold("Outcome  ") + style.yellow("STUCK"))
    else:
        out.append(style.bold(f"Outcome  {terminal}"))
    out.append(style.bold(rule))
    out.append("")

    if answer_text:
        for paragraph in answer_text.split("\n"):
            if not paragraph.strip():
                out.append("")
                continue
            for line in textwrap.wrap(paragraph, width=width - 2) or [""]:
                out.append(f"  {line}")
        out.append("")
    if stuck_reason:
        out.append(style.yellow("  stuck reason:"))
        for paragraph in stuck_reason.split("\n"):
            if not paragraph.strip():
                out.append("")
                continue
            for line in textwrap.wrap(paragraph, width=width - 4) or [""]:
                out.append(f"    {line}")
        out.append("")

    if facts:
        out.append(f"  Confirmed facts ({len(facts)}):")
        for f in facts:
            conf = style.confidence(f.get("confidence"))
            claim = f.get("claim", "")
            wrapped = textwrap.wrap(
                f"[{conf}] {claim}",
                width=width - 4,
                initial_indent="    ",
                subsequent_indent="           ",
            )
            for w in wrapped:
                out.append(w)
    if hyps:
        out.append("")
        out.append(f"  Working hypotheses ({len(hyps)}):")
        for h in hyps:
            claim = h.get("claim", "")
            for w in textwrap.wrap(claim, width=width - 6,
                                   initial_indent="    - ",
                                   subsequent_indent="      "):
                out.append(w)
    if dead_ends:
        out.append("")
        out.append(f"  Dead ends ({len(dead_ends)}):")
        for d in dead_ends:
            desc = d.get("description", "")
            for w in textwrap.wrap(desc, width=width - 6,
                                   initial_indent="    - ",
                                   subsequent_indent="      "):
                out.append(w)

    return out


def _render_summary(
    events: list[dict[str, Any]],
    turns: list[Turn],
    *,
    style: Style,
    width: int,
) -> list[str]:
    out: list[str] = []
    n_dispatches = sum(1 for e in events if e.get("kind") == "after_dispatch") + sum(
        1 for e in events if e.get("kind") == "on_error"
    )
    n_errors = sum(1 for e in events if e.get("kind") == "on_error")
    # Wall time from first to last event timestamp.
    timestamps = [e.get("timestamp") for e in events if e.get("timestamp") is not None]
    wall_seconds = (
        timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else None
    )

    # Tool usage counts.
    from collections import Counter

    tool_counts: Counter[str] = Counter(
        e["tool"] for e in events
        if e.get("kind") == "before_dispatch" and e.get("tool")
    )

    out.append(f"  Turns:       {len(turns)}")
    err_str = f"  ({n_errors} errors)" if n_errors else ""
    out.append(f"  Dispatches:  {n_dispatches}{err_str}")
    if wall_seconds is not None:
        out.append(f"  Wall time:   {wall_seconds:.1f} s")
    if tool_counts:
        tools_summary = ", ".join(f"{t}×{n}" for t, n in tool_counts.most_common())
        for w in textwrap.wrap(
            tools_summary,
            width=width - 16,
            initial_indent="  Tools used:  ",
            subsequent_indent="               ",
        ):
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_call(tool: str, args: dict[str, Any], *, style: Style) -> str:
    """Compact one-line tool-call rendering.

    Long string arguments are truncated. SQL queries get a "SELECT…" stub
    rather than the full statement, which we'd see in full on the EXEC line
    if needed.
    """
    inner = []
    for k, v in args.items():
        v_str = _short_arg(k, v)
        inner.append(f"{k}={v_str}")
    return f"{style.cyan(tool)}({', '.join(inner)})"


def _short_arg(key: str, value: Any) -> str:
    if isinstance(value, str):
        if key == "query":
            # SQL — show just the first verb to keep the line short.
            head = value.strip().split()[:3]
            return repr(" ".join(head) + ("…" if len(value) > 40 else ""))
        if len(value) > 30:
            return repr(value[:27] + "…")
        return repr(value)
    return repr(value)


def _format_result_summary(rs: dict[str, Any], *, style: Style) -> str:
    if "set_handle" in rs:
        return f"{style.yellow(rs['set_handle'])} size={rs.get('set_size')}"
    vt = rs.get("value_type")
    if vt == "list":
        return f"list[{rs.get('length')}]"
    if vt == "dict":
        return "dict(" + ", ".join(rs.get("keys") or []) + ")"
    if "value" in rs:
        return repr(rs["value"])
    return repr(rs)


def _decision_colour(decision: str, style: Style) -> str:
    table = {
        "answer": style.green,
        "stuck": style.yellow,
        "continue": style.cyan,
        "revise_plan": style.yellow,
        "checkpoint": style.yellow,
    }
    fn = table.get(decision, style.dim)
    return fn(decision)


def _first_event(events: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for e in events:
        if e.get("kind") == kind:
            return e
    return None


# ---------------------------------------------------------------------------
# Trace resolution & CLI
# ---------------------------------------------------------------------------


def resolve_trace_path(spec: str | None, traces_dir: Path) -> Path:
    """Turn a user-supplied spec (filename, session id, or None) into a Path."""
    if spec is None:
        return latest_trace(traces_dir)
    p = Path(spec)
    if p.exists():
        return p
    # Maybe a session id?
    if SESSION_ID_RE.match(spec):
        candidate = traces_dir / f"{spec}.jsonl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {spec!r} to a trace file. "
        f"Looked at {p} and {traces_dir / (spec + '.jsonl')}."
    )


def latest_trace(traces_dir: Path) -> Path:
    files = sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            f"No trace files found in {traces_dir}."
        )
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ancestors.trace_viewer",
        description="Pretty-print a session trace.",
    )
    parser.add_argument(
        "spec",
        nargs="?",
        help="Trace filename or 12-char session id. Defaults to the latest "
             "trace in --traces-dir.",
    )
    parser.add_argument("--traces-dir", default=str(DEFAULT_TRACES_DIR), type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour. Auto-disabled when stdout "
                             "is not a TTY.")
    parser.add_argument("--width", type=int, default=80)
    args = parser.parse_args(argv)

    use_color = not args.no_color and sys.stdout.isatty()
    style = Style(enabled=use_color)

    try:
        path = resolve_trace_path(args.spec, args.traces_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    events = load_trace(path)
    print(render(events, style=style, width=args.width,
                 summary_only=args.summary_only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
