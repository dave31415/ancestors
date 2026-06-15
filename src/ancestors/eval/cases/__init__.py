"""Auto-discover Case lists from sibling modules.

Each module exports `CASES: list[Case]`. `all_cases()` aggregates them
and assigns the `suite` field based on the module name.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ancestors.eval.case import Case


def all_cases() -> list[Case]:
    """Walk every sibling module and collect its CASES."""
    from ancestors.eval.case import Case as _Case

    cases: list[_Case] = []
    package = __name__
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{package}.{info.name}")
        for case in getattr(module, "CASES", []):
            cases.append(case)
    return cases


def cases_in_suite(suite: str) -> list[Case]:
    return [c for c in all_cases() if c.suite == suite]


def case_by_name(fqname: str) -> Case | None:
    for c in all_cases():
        if c.fully_qualified_name == fqname or c.name == fqname:
            return c
    return None
