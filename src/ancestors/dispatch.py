"""Safe tool dispatch — the only execution path agents are allowed.

Design goals (in priority order):

  1. The agent cannot execute arbitrary code. It dispatches by name.
  2. The agent cannot fabricate inputs. Free strings are length-bounded,
     ids are regex-constrained, IdSets are referenced by opaque handles
     produced by prior calls — the agent never sees raw id lists.
  3. The agent cannot exceed configured ceilings. Set sizes, hydration
     limits, generation depths are validated before tool execution and
     enforced by the registered Pydantic argument schemas.
  4. Tools cannot escape their domain. They are pure functions over the
     in-memory GedcomDatabase. No file IO, no subprocess, no network.
  5. The validator is plain Python with no LLM involvement. Even if the
     model layer is fully compromised, every call still passes through
     deterministic schema + bounds checks before any tool function runs.

The dispatcher returns structured results — successful or failed — but
never raises. Tool exceptions are caught and converted to ToolError so the
agent loop's contract is uniform: every dispatch returns either `result`
or `error`.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field, StringConstraints, ValidationError

from ancestors import dsl
from ancestors.agent.observability import HookRegistry
from ancestors.models import (
    EventType,
    GapType,
    IdSet,
    Individual,
    Sex,
    ToolError,
)

log = logging.getLogger(__name__)

# ---- Hard limits enforced by the dispatcher, not the tool ----
MAX_INPUT_SET_SIZE = 50_000
MAX_RESULT_SET_SIZE = 50_000
MAX_HYDRATION_LIMIT = 100
MAX_GENERATIONS = 30
MAX_STRING_ARG_LEN = 200
MAX_CALLS_PER_SESSION = 500

PERSON_ID_RE = re.compile(r"^@[A-Z][A-Z0-9_\-]{0,200}@$")

PersonId = Annotated[str, StringConstraints(pattern=PERSON_ID_RE.pattern, max_length=204)]
SetHandle = Annotated[str, StringConstraints(pattern=r"^h_[0-9]+$", max_length=16)]
BoundedStr = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_STRING_ARG_LEN)
]
Generations = Annotated[int, Field(ge=0, le=MAX_GENERATIONS)]
HydrationLimit = Annotated[int, Field(ge=1, le=MAX_HYDRATION_LIMIT)]


# ---------------------------------------------------------------------------
# Per-tool argument schemas. Every tool has one, even if it takes no args.
# These ARE the agent-facing input contract.
# ---------------------------------------------------------------------------


class _NoArgs(BaseModel):
    model_config = {"extra": "forbid"}


class _PersonOnly(BaseModel):
    model_config = {"extra": "forbid"}
    person_id: PersonId


class _AncestorsArgs(BaseModel):
    model_config = {"extra": "forbid"}
    person_id: PersonId
    max_generations: Generations = 20


class _SetOnly(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle


class _TwoSets(BaseModel):
    model_config = {"extra": "forbid"}
    a: SetHandle
    b: SetHandle


class _FilterSurname(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    surname: BoundedStr


class _FilterGiven(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    fragment: BoundedStr


class _FilterSex(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    sex: Sex


class _FilterEventPlace(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    event_type: EventType
    place_contains: BoundedStr


class _FilterEventYearRange(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    event_type: EventType
    year_min: int | None = Field(default=None, ge=1, le=9999)
    year_max: int | None = Field(default=None, ge=1, le=9999)


class _FilterEventTypeOnly(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    event_type: EventType


class _Hydrate(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    limit: HydrationLimit = 25


class _GroupBy(BaseModel):
    model_config = {"extra": "forbid"}
    ids: SetHandle
    by: Annotated[str, StringConstraints(pattern=r"^(generation|birth_century|birth_country|sex)$")]


# ---------------------------------------------------------------------------
# Registry — the whitelist. Only tools listed here are dispatchable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    description: str


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("all_individuals", _NoArgs, dsl.all_individuals,
             "Return an IdSet of every individual in the database."),
        Tool("get_ancestors_of", _AncestorsArgs, dsl.get_ancestors_of,
             "Ancestors of person_id, root included, with generation metadata."),
        Tool("get_descendants_of", _AncestorsArgs, dsl.get_descendants_of,
             "Descendants of person_id, root included, with generation metadata."),
        Tool("get_parents_of", _PersonOnly, dsl.get_parents_of,
             "Direct parents (husband+wife of person's child-family)."),
        Tool("get_children_of", _PersonOnly, dsl.get_children_of,
             "Direct children across all spouse-families."),
        Tool("get_siblings_of", _PersonOnly, dsl.get_siblings_of,
             "Direct siblings (children of the same parent-family)."),
        Tool("get_spouses_of", _PersonOnly, dsl.get_spouses_of,
             "All recorded spouses across spouse-families."),
        Tool("intersect", _TwoSets, dsl.intersect,
             "Set intersection. Metadata from a wins on overlap."),
        Tool("union", _TwoSets, dsl.union,
             "Set union, preserving first-seen order."),
        Tool("difference", _TwoSets, dsl.difference,
             "Elements of a not in b."),
        Tool("filter_by_surname", _FilterSurname, dsl.filter_by_surname,
             "Keep only ids whose primary name surname matches exactly."),
        Tool("filter_by_given_name_contains", _FilterGiven, dsl.filter_by_given_name_contains,
             "Keep only ids whose given name contains the fragment."),
        Tool("filter_by_sex", _FilterSex, dsl.filter_by_sex,
             "Keep only ids matching the specified sex."),
        Tool("filter_by_event_place", _FilterEventPlace, dsl.filter_by_event_place,
             "Keep ids with at least one event of type T whose place mentions the string."),
        Tool("filter_by_event_year_range", _FilterEventYearRange, dsl.filter_by_event_year_range,
             "Keep ids with at least one event of type T in the given year range."),
        Tool("filter_has_no_source", _FilterEventTypeOnly, dsl.filter_has_no_source,
             "Keep ids whose event of type T exists but has no source citation."),
        Tool("count", _SetOnly, dsl.count,
             "Return the size of an IdSet."),
        Tool("group_by", _GroupBy, dsl.group_by,
             "Aggregate counts by generation/birth_century/birth_country/sex."),
        Tool("get_individuals", _Hydrate, dsl.get_individuals,
             "Hydrate up to `limit` ids into full Individual records."),
        Tool("get_summary", _PersonOnly, dsl.get_summary,
             "Compact, deterministic one-line factual précis of a person."),
        Tool("get_evidence_gaps", _PersonOnly, dsl.get_evidence_gaps,
             "List evidence-gap types for a person (driving the research agenda)."),
    ]
}


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

    Observability: when a HookRegistry is attached, every call emits
    `before_dispatch`, then either `after_dispatch` or `on_error`, with
    timing data. The hook layer never affects the result — observer
    exceptions are logged and swallowed.
    """

    sets: dict[str, IdSet] = field(default_factory=dict)
    call_count: int = 0
    hooks: HookRegistry | None = None
    _next_handle: int = 1

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

        tool = TOOLS.get(tool_name)
        if tool is None:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                error=ToolError(
                    code="unknown_tool",
                    message=f"Tool {tool_name!r} is not registered.",
                ),
            )

        try:
            validated = tool.args_model.model_validate(args)
        except ValidationError as exc:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                error=ToolError(
                    code="invalid_arguments",
                    message="Arguments failed schema validation.",
                    detail={"errors": exc.errors(include_url=False)},
                ),
            )

        try:
            kwargs = self._resolve_args(validated)
        except KeyError as exc:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                error=ToolError(
                    code="unknown_set_handle",
                    message=f"Set handle {exc.args[0]!r} not known in this session.",
                ),
            )
        except ValueError as exc:
            return DispatchResult(
                ok=False,
                tool=tool_name,
                error=ToolError(code="input_too_large", message=str(exc)),
            )

        try:
            result = tool.fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 — converted to structured error
            log.exception("Tool %s raised an unexpected exception", tool_name)
            return DispatchResult(
                ok=False,
                tool=tool_name,
                error=ToolError(
                    code="tool_exception",
                    message=f"{type(exc).__name__}: {exc}",
                ),
            )

        return self._envelope(tool_name, result)

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

    def _envelope(self, tool_name: str, result: Any) -> DispatchResult:
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
            return DispatchResult(
                ok=True, tool=tool_name, set_handle=handle, set_size=result.size
            )
        if isinstance(result, list) and result and isinstance(result[0], Individual):
            return DispatchResult(
                ok=True, tool=tool_name, value=[ind.model_dump() for ind in result]
            )
        if isinstance(result, list) and result and isinstance(result[0], GapType):
            return DispatchResult(
                ok=True, tool=tool_name, value=[g.value for g in result]
            )
        return DispatchResult(ok=True, tool=tool_name, value=result)


def list_tools() -> list[dict[str, str]]:
    """Schema-level introspection — what the agent would be told it can do."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "args_schema": str(t.args_model.model_json_schema()),
        }
        for t in TOOLS.values()
    ]


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
