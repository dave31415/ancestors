"""Demonstrate three different research questions answered through dispatch only.

The script simulates an agent: every "step" is a single dispatch call carrying
nothing but a tool name and a JSON-friendly dict of arguments. No DSL function
is imported. No Python code crosses the dispatcher boundary. An LLM emitting
these same tool calls — even a fully compromised one — could not execute
anything outside the registered surface.

Run with:
    uv run python demos/run_queries.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ancestors.dispatch import Dispatcher
from ancestors.session import bind_session
from ancestors.tools.gedcom import load_gedcom

GEDCOM_PATH = Path(__file__).resolve().parent.parent / "data" / "export-Ancestors.ged"
DAVID_ID = "@I6000000001904015159@"


def call(d: Dispatcher, tool: str, args: dict) -> dict:
    """Print the agent's tool call + the dispatcher's envelope, return it as dict."""
    result = d.dispatch(tool, args)
    envelope = result.model_dump(exclude_none=True)
    # Trim value lists for readability — the contract is still the same.
    if "value" in envelope and isinstance(envelope["value"], list):
        envelope["value"] = envelope["value"][:3] + (["…"] if len(envelope["value"]) > 3 else [])
    print(f"  → dispatch({tool!r}, {json.dumps(args)})")
    print(f"    {json.dumps(envelope, default=str)}")
    return envelope


def heading(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    bind_session(load_gedcom(GEDCOM_PATH))
    d = Dispatcher()

    # -----------------------------------------------------------------------
    # Q1: Did any of David's ancestors come from England?
    # -----------------------------------------------------------------------
    heading("Q1: Did any of David's ancestors come from England?")

    a = call(d, "get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 25})
    b = call(d, "filter_by_event_place", {
        "ids": a["set_handle"],
        "event_type": "BIRT",
        "place_contains": "England",
    })
    c = call(d, "count", {"ids": b["set_handle"]})
    s = call(d, "get_individuals", {"ids": b["set_handle"], "limit": 3})

    print(f"\n  ANSWER: {c['value']} ancestors have a recorded birth in England.")
    print("  Examples:")
    for ind in s["value"][:3]:
        name = ind["names"][0]["full"] if ind.get("names") else "?"
        place = ind.get("birth", {}).get("place", {}) or {}
        loc = ", ".join(p for p in (place.get("city"), place.get("state"), place.get("country")) if p)
        print(f"    - {name} ({loc})")

    # -----------------------------------------------------------------------
    # Q2: Among David's ancestors, how many have an unsourced birth record?
    #     (Reuses get_ancestors_of, plus a different refiner.)
    # -----------------------------------------------------------------------
    heading("Q2: How many of David's ancestors have an unsourced birth record?")

    # We can reuse `a` from Q1 — but the agent would re-call to be self-contained.
    a2 = call(d, "get_ancestors_of", {"person_id": DAVID_ID, "max_generations": 25})
    u = call(d, "filter_has_no_source", {
        "ids": a2["set_handle"],
        "event_type": "BIRT",
    })
    n = call(d, "count", {"ids": u["set_handle"]})
    g = call(d, "group_by", {"ids": u["set_handle"], "by": "birth_century"})

    print(f"\n  ANSWER: {n['value']} ancestors with a recorded birth carry no source citation.")
    print(f"  Distribution by century: {g['value']}")

    # -----------------------------------------------------------------------
    # Q3: Are there any Gallaghers born 1830–1900 with no source on birth?
    #     (Reuses filter_by_event_year_range and filter_has_no_source.)
    # -----------------------------------------------------------------------
    heading("Q3: Are there any Gallaghers born 1830-1900 with no source on birth?")

    everyone = call(d, "all_individuals", {})
    gal = call(d, "filter_by_surname", {"ids": everyone["set_handle"], "surname": "Gallagher"})
    in_range = call(d, "filter_by_event_year_range", {
        "ids": gal["set_handle"],
        "event_type": "BIRT",
        "year_min": 1830,
        "year_max": 1900,
    })
    no_src = call(d, "filter_has_no_source", {
        "ids": in_range["set_handle"],
        "event_type": "BIRT",
    })
    cnt = call(d, "count", {"ids": no_src["set_handle"]})
    inds = call(d, "get_individuals", {"ids": no_src["set_handle"], "limit": 5})

    print(f"\n  ANSWER: {cnt['value']} Gallagher(s) born 1830-1900 with no source on birth.")
    for ind in inds["value"][:5]:
        name = ind["names"][0]["full"] if ind.get("names") else "?"
        year = (ind.get("birth") or {}).get("date", {}).get("year")
        print(f"    - {name} (b. {year})")

    heading("Dispatcher trace summary")
    print(f"  Total calls executed: {d.call_count}")
    print(f"  IdSets stored in session: {len(d.sets)}")
    print(f"  Set sizes: {[s.size for s in d.sets.values()]}")


if __name__ == "__main__":
    main()
