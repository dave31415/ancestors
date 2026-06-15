"""Core eval types.

A `Case` is a question + a list of checks. Running it produces a
`CaseResult` — the loop's result, the timing, and one `CheckResult`
per check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ancestors.agent.loop import LoopResult
from ancestors.agent.observability import TraceEvent


@dataclass
class CheckResult:
    """One check's outcome on a case."""

    passed: bool
    message: str
    name: str = "<unnamed>"


# A check takes the loop result + the captured trace events and returns
# a CheckResult. Factories in `checks.py` produce these.
Check = Callable[[LoopResult, list[TraceEvent]], CheckResult]


@dataclass(frozen=True)
class Case:
    """An eval question with its checks.

    Tied to a suite (the cases/<suite>.py file it came from) via the
    `suite` field. The `fully_qualified_name` ("suite/name") is what shows
    up in the baseline JSON and the scorecard.
    """

    name: str
    suite: str
    question: str
    checks: list[Check] = field(default_factory=list)

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.suite}/{self.name}"


@dataclass
class CaseResult:
    """The outcome of running one case."""

    case: Case
    loop_result: LoopResult
    check_results: list[CheckResult]
    wall_seconds: float
    dispatch_count: int
    llm_call_count: int
    tools_used: list[str]

    @property
    def passed(self) -> bool:
        """A case passes only if every check passes."""
        return all(c.passed for c in self.check_results)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.check_results if not c.passed]

    @property
    def status_label(self) -> str:
        """Three-state label for scorecards: PASS / FAIL / STUCK."""
        terminal = self.loop_result.terminal_state.value
        if terminal == "STUCK":
            return "STUCK"
        return "PASS" if self.passed else "FAIL"
