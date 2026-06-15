"""Safe tool dispatch — the only execution path agents are allowed.

Design goals (in priority order):

  1. The agent cannot execute arbitrary code. It dispatches by name.
  2. The agent cannot fabricate inputs. Free strings are length-bounded,
     ids are regex-constrained, IdSets are referenced by opaque handles
     produced by prior calls — the agent never sees raw id lists.
  3. The agent cannot exceed configured ceilings. Set sizes, hydration
     limits, generation depths are validated before tool execution and
     enforced by the registered Pydantic argument schemas.
  4. Tools cannot escape their domain. They are expected to be pure
     functions over in-memory state — no file IO, no subprocess, no
     network. The dispatcher cannot prove this property; it's a
     convention domain authors enforce in the registered tool functions.
  5. The validator is plain Python with no LLM involvement. Even if the
     model layer is fully compromised, every call still passes through
     deterministic schema + bounds checks before any tool function runs.

The dispatcher returns structured results — successful or failed — but
never raises. Tool exceptions are caught and converted to ToolError so the
agent loop's contract is uniform: every dispatch returns either `result`
or `error`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ancestors.agent.observability import HookRegistry
from ancestors.models import IdSet, ToolError

log = logging.getLogger(__name__)

# ---- Hard limits enforced by the dispatcher, not the tool ----
# These are the generic ceilings the engine enforces regardless of which
# tool runs. Per-tool argument bounds (id formats, string length, generation
# depth) live with the tool registry on the domain side.
MAX_INPUT_SET_SIZE = 50_000
MAX_RESULT_SET_SIZE = 50_000
MAX_HYDRATION_LIMIT = 100
MAX_STRING_ARG_LEN = 200
MAX_CALLS_PER_SESSION = 500


# ---------------------------------------------------------------------------
# Tool — what the dispatcher operates on. A domain populates a registry of
# these (see ancestors/tool_registry.py for the genealogy version) and
# hands it to the Dispatcher.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    description: str


# ---------------------------------------------------------------------------
# Dispatcher — single entry point.
# ---------------------------------------------------------------------------


class DispatchResult(BaseModel):
    """The uniform envelope returned to the agent for every call."""

    ok: bool
    tool: str
    set_handle: str | None = None
    set_size: int | None = None
    value: Any | None = None
    error: ToolError | None = None


@dataclass
class Dispatcher:
    """Stateful per-session dispatcher.

    Holds the IdSet handle store and call count. Tools never see this object
    — they see only their declared arguments, with IdSet inputs resolved
    transparently by `_resolve_args`.

    The `tools` registry is injected by the caller (typically derived from
    the active Corpus). The dispatcher is otherwise domain-agnostic: it
    enforces the call budget, validates arguments against each Tool's
    declared Pydantic schema, resolves set handles, runs the function,
    and wraps the result in a uniform DispatchResult envelope.

    Observability: when a HookRegistry is attached, every call emits
    `before_dispatch`, then either `after_dispatch` or `on_error`, with
    timing data. The hook layer never affects the result — observer
    exceptions are logged and swallowed.
    """

    tools: dict[str, Tool] = field(default_factory=dict)
    sets: dict[str, IdSet] = field(default_factory=dict)
    call_count: int = 0
    hooks: HookRegistry | None = None
    _next_handle: int = 1
    _memo: dict[str, DispatchResult] = field(default_factory=dict)
    # Provenance: handle -> (tool_name, args_dict). Used by handles_summary()
    # so the agent's prompt can show what each handle represents.
    _handle_provenance: dict[str, tuple[str, dict]] = field(default_factory=dict)

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> DispatchResult:
        start = time.time()
        if self.hooks is not None:
            self.hooks.emit("before_dispatch", tool=tool_name, args=args)
        result = self._dispatch_impl(tool_name, args)
        duration_ms = (time.time() - start) * 1000
        if self.hooks is not None:
            if result.ok:
                self.hooks.emit(
                    "after_dispatch",
                    tool=tool_name,
                    args=args,
                    result_summary=_summarize_result(result),
                    duration_ms=duration_ms,
                )
            else:
                self.hooks.emit(
                    "on_error",
                    tool=tool_name,
                    args=args,
                    error_code=result.error.code if result.error else None,
                    error_message=result.error.message if result.error else None,
                    duration_ms=duration_ms,
                )
        return result

    def _dispatch_impl(self, tool_name: str, args: dict[str, Any]) -> DispatchResult:
        if self.call_count >= MAX_CALLS_PER_SESSION:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                error=ToolError(
                    code="call_budget_exceeded",
                    message=f"Session call budget {MAX_CALLS_PER_SESSION} reached.",
                ),
            )
        self.call_count += 1

        # Memoization: identical (tool, args) returns the prior result
        # rather than dispatching again. Cheap, deterministic, and prevents
        # the wasted-call class of agent friction at the source. Counted
        # toward the call budget so the dispatcher cap still bites if an
        # agent loops; the stall detector at the loop level still bites too.
        memo_key = _memo_key(tool_name, args)
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached

        tool = self.tools.get(tool_name)
        if tool is None:
            return self._cache_and_return(
                memo_key,
                DispatchResult(
                    ok=False,
                    tool=tool_name,
                    error=ToolError(
                        code="unknown_tool",
                        message=f"Tool {tool_name!r} is not registered.",
                    ),
                ),
            )

        try:
            validated = tool.args_model.model_validate(args)
        except ValidationError as exc:
            return self._cache_and_return(
                memo_key,
                DispatchResult(
                    ok=False,
                    tool=tool_name,
                    error=ToolError(
                        code="invalid_arguments",
                        message=_format_validation_errors(exc, tool.args_model),
                        detail={"errors": exc.errors(include_url=False)},
                    ),
                ),
            )

        try:
            kwargs = self._resolve_args(validated)
        except KeyError as exc:
            return self._cache_and_return(
                memo_key,
                DispatchResult(
                    ok=False,
                    tool=tool_name,
                    error=ToolError(
                        code="unknown_set_handle",
                        message=f"Set handle {exc.args[0]!r} not known in this session.",
                    ),
                ),
            )
        except ValueError as exc:
            return self._cache_and_return(
                memo_key,
                DispatchResult(
                    ok=False,
                    tool=tool_name,
                    error=ToolError(code="input_too_large", message=str(exc)),
                ),
            )

        try:
            result = tool.fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 — converted to structured error
            log.exception("Tool %s raised an unexpected exception", tool_name)
            return self._cache_and_return(
                memo_key,
                DispatchResult(
                    ok=False,
                    tool=tool_name,
                    error=ToolError(
                        code="tool_exception",
                        message=f"{type(exc).__name__}: {exc}",
                    ),
                ),
            )

        envelope = self._envelope(tool_name, args, result)
        return self._cache_and_return(memo_key, envelope)

    def _cache_and_return(
        self, memo_key: str, result: DispatchResult
    ) -> DispatchResult:
        self._memo[memo_key] = result
        return result

    def get_set(self, handle: str) -> IdSet:
        """Internal accessor used by demo code; never exposed to agents."""
        return self.sets[handle]

    def _resolve_args(self, validated: BaseModel) -> dict[str, Any]:
        """Convert SetHandle fields to IdSet objects from the handle store.

        Any field declared as a SetHandle by the args model is looked up here.
        Validation has already enforced the `^h_\\d+$` pattern, so a missing
        handle is a stale or malicious reference rather than a typo.
        """
        kwargs: dict[str, Any] = {}
        for name, value in validated.model_dump().items():
            if isinstance(value, str) and value.startswith("h_") and value[2:].isdigit():
                if value not in self.sets:
                    raise KeyError(value)
                resolved = self.sets[value]
                if resolved.size > MAX_INPUT_SET_SIZE:
                    raise ValueError(
                        f"Input set {value} too large ({resolved.size} > {MAX_INPUT_SET_SIZE})"
                    )
                kwargs[name] = resolved
            else:
                kwargs[name] = value
        return kwargs

    def _envelope(
        self, tool_name: str, args: dict[str, Any], result: Any
    ) -> DispatchResult:
        if isinstance(result, IdSet):
            if result.size > MAX_RESULT_SET_SIZE:
                return DispatchResult(
                    ok=False,
                    tool=tool_name,
                    error=ToolError(
                        code="result_too_large",
                        message=f"Result set exceeds ceiling ({result.size} > {MAX_RESULT_SET_SIZE}).",
                    ),
                )
            handle = f"h_{self._next_handle}"
            self._next_handle += 1
            self.sets[handle] = result
            self._handle_provenance[handle] = (tool_name, dict(args))
            return DispatchResult(
                ok=True, tool=tool_name, set_handle=handle, set_size=result.size
            )
        if isinstance(result, list) and result:
            if isinstance(result[0], BaseModel):
                return DispatchResult(
                    ok=True,
                    tool=tool_name,
                    value=[m.model_dump() for m in result],
                )
            if isinstance(result[0], Enum):
                return DispatchResult(
                    ok=True,
                    tool=tool_name,
                    value=[e.value for e in result],
                )
        if isinstance(result, BaseModel):
            # Any structured Pydantic result (CommonAncestorResult, future
            # scalar-shaped tools) is rendered to a dict the LLM can read.
            return DispatchResult(ok=True, tool=tool_name, value=result.model_dump())
        return DispatchResult(ok=True, tool=tool_name, value=result)

    def handles_summary(self) -> str:
        """A compact rendering of every IdSet handle currently in the session.

        Goes into the agent's prompt at PLAN/ASSESS time so the model can
        reuse handles it has already produced instead of re-creating them.
        Combined with memoization, this is the load-bearing fix for the
        repeat-call pathology we observed.
        """
        if not self._handle_provenance:
            return "(no IdSet handles in this session yet)"
        lines: list[str] = []
        for handle, (tool, args) in self._handle_provenance.items():
            size = self.sets[handle].size if handle in self.sets else "?"
            args_str = _short_args(args)
            lines.append(f"  {handle} (size {size}) ← {tool}({args_str})")
        return "\n".join(lines)


def _memo_key(tool_name: str, args: dict[str, Any]) -> str:
    """Canonical dispatch key. Same tool + same args → same string."""
    return f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"


def _short_args(args: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in args.items():
        s = json.dumps(v, default=str)
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _format_validation_errors(
    exc: ValidationError, args_model: type[BaseModel]
) -> str:
    """Render a Pydantic validation error as one actionable sentence.

    The default Pydantic message lists each error with type tags that aren't
    very model-readable. Here we name the field, the problem, and (for
    enums and literals) the allowed values, so the agent can usually fix
    its call on the next turn rather than guessing.
    """
    schema = args_model.model_json_schema()
    props = schema.get("properties", {})
    defs = schema.get("$defs", {})
    required = set(schema.get("required", []))

    def field_schema(field_name: str) -> dict[str, Any]:
        node = props.get(field_name, {})
        if "$ref" in node and node["$ref"].startswith("#/$defs/"):
            return defs.get(node["$ref"].split("/")[-1], node)
        return node

    fragments: list[str] = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        etype = err.get("type", "")
        msg = err.get("msg", "validation failed")
        ctx = err.get("ctx") or {}

        if etype == "missing":
            allowed = field_schema(loc).get("enum")
            if allowed:
                fragments.append(
                    f"required field {loc!r} is missing — must be one of {allowed}"
                )
            else:
                fragments.append(f"required field {loc!r} is missing")
        elif etype in {"literal_error", "enum"}:
            expected = ctx.get("expected") or field_schema(loc).get("enum")
            fragments.append(
                f"field {loc!r}: invalid value; must be one of {expected}"
            )
        elif etype == "string_pattern_mismatch":
            pattern = ctx.get("pattern", "?")
            fragments.append(
                f"field {loc!r}: {msg} (pattern {pattern})"
            )
        elif etype.startswith("string_too_"):
            limit = ctx.get("max_length") or ctx.get("min_length")
            fragments.append(f"field {loc!r}: {msg} (limit {limit})")
        elif etype in {"greater_than_equal", "less_than_equal", "greater_than", "less_than"}:
            fragments.append(f"field {loc!r}: {msg}")
        elif loc == "(root)" and etype == "extra_forbidden":
            fragments.append(
                f"unexpected field passed to this tool"
            )
        else:
            fragments.append(f"field {loc!r}: {msg}")

    body = "; ".join(fragments) if fragments else "validation failed"
    hint = ""
    if required:
        hint = f"  (this tool requires: {sorted(required)})"
    return f"{body}.{hint}".strip()


def _summarize_result(result: DispatchResult) -> dict[str, Any]:
    """Compact, observable view of a successful result.

    Keeps trace lines small — we log handle + size for sets, value type +
    length for hydration, raw value for scalars. Full payloads stay in the
    dispatcher's set store, off the trace.
    """
    if result.set_handle is not None:
        return {"set_handle": result.set_handle, "set_size": result.set_size}
    value = result.value
    if isinstance(value, list):
        return {"value_type": "list", "length": len(value)}
    if isinstance(value, dict):
        return {"value_type": "dict", "keys": list(value.keys())[:8]}
    return {"value_type": type(value).__name__, "value": value}
