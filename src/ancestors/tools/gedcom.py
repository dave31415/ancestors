"""GEDCOM parsing tools.

Wraps ged4py to produce our domain models. ged4py types are confined to this
module — every public function returns a Pydantic model from `ancestors.models`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ged4py.parser import GedcomReader

from ancestors.models import (
    Event,
    EventType,
    Family,
    GedcomDatabase,
    GedcomDate,
    Individual,
    Name,
    Place,
    Sex,
)

log = logging.getLogger(__name__)

# ged4py expands GEDCOM date keywords (ABT → ABOUT, BEF → BEFORE, etc.) when
# stringifying. We accept both forms so this works against any GEDCOM source.
_APPROX_PREFIXES = (
    "ABT", "ABOUT",
    "EST", "ESTIMATED",
    "CAL", "CALCULATED",
    "BEF", "BEFORE",
    "AFT", "AFTER",
    "BET", "BETWEEN",
    "FROM", "TO",
)
_YEAR_RE = re.compile(r"\b(\d{4})\b")

_EVENT_TAGS: dict[str, EventType] = {
    "BIRT": EventType.BIRTH,
    "DEAT": EventType.DEATH,
    "BURI": EventType.BURIAL,
    "MARR": EventType.MARRIAGE,
    "DIV": EventType.DIVORCE,
    "BAPM": EventType.BAPTISM,
    "IMMI": EventType.IMMIGRATION,
    "EMIG": EventType.EMIGRATION,
    "CENS": EventType.CENSUS,
    "OCCU": EventType.OCCUPATION,
    "RESI": EventType.RESIDENCE,
}


class GedcomLoadError(Exception):
    """Raised when a GEDCOM file cannot be opened or parsed."""


class IndividualNotFoundError(KeyError):
    """Raised when an individual ID is not present in the database."""


class FamilyNotFoundError(KeyError):
    """Raised when a family ID is not present in the database."""


def load_gedcom(path: str | Path) -> GedcomDatabase:
    """Parse a GEDCOM file and return an in-memory queryable database."""
    path = Path(path)
    if not path.exists():
        raise GedcomLoadError(f"GEDCOM file not found: {path}")

    log.info("Loading GEDCOM file: %s", path)
    individuals: dict[str, Individual] = {}
    families: dict[str, Family] = {}

    try:
        with GedcomReader(str(path)) as reader:
            for rec in reader.records0("INDI"):
                individual = _build_individual(rec)
                individuals[individual.id] = individual
            for rec in reader.records0("FAM"):
                family = _build_family(rec)
                families[family.id] = family
    except GedcomLoadError:
        raise
    except Exception as exc:
        raise GedcomLoadError(f"Failed to parse {path}: {exc}") from exc

    log.info(
        "Loaded %d individuals and %d families from %s",
        len(individuals),
        len(families),
        path,
    )
    return GedcomDatabase(
        source_path=str(path.resolve()),
        individuals=individuals,
        families=families,
    )


def get_individual(db: GedcomDatabase, id: str) -> Individual:
    """Look up a single individual by GEDCOM xref id (e.g. '@I1126034@')."""
    try:
        return db.individuals[id]
    except KeyError:
        raise IndividualNotFoundError(id) from None


def get_family(db: GedcomDatabase, id: str) -> Family:
    """Look up a family by GEDCOM xref id (e.g. '@F4025896807070010511@').

    The Family object carries husband/wife/child xref ids — callers can
    resolve them against db.individuals as needed. We deliberately do not
    return an inflated structure here: keeping the tool a simple lookup
    means tree traversal stays explicit in the lineage tools.
    """
    try:
        return db.families[id]
    except KeyError:
        raise FamilyNotFoundError(id) from None


def _build_individual(rec: Any) -> Individual:
    names = list(_collect_names(rec))
    sex = _parse_sex(_first_value(rec, "SEX"))

    birth = _build_event_from_tag(rec, "BIRT")
    death = _build_event_from_tag(rec, "DEAT")
    burial = _build_event_from_tag(rec, "BURI")

    other_events: list[Event] = []
    for sub in rec.sub_records:
        if sub.tag in {"BIRT", "DEAT", "BURI"}:
            continue
        if sub.tag in _EVENT_TAGS:
            event = _build_event(sub, _EVENT_TAGS[sub.tag])
            if event is not None:
                other_events.append(event)

    return Individual(
        id=rec.xref_id,
        names=names or [Name(full=str(rec.name))],
        sex=sex,
        birth=birth,
        death=death,
        burial=burial,
        other_events=other_events,
        notes=[s.value for s in _subs(rec, "NOTE") if s.value],
        families_as_child=[s.value for s in _subs(rec, "FAMC") if s.value],
        families_as_spouse=[s.value for s in _subs(rec, "FAMS") if s.value],
        source_ids=[s.value for s in _subs(rec, "SOUR") if s.value],
    )


def _build_family(rec: Any) -> Family:
    return Family(
        id=rec.xref_id,
        husband_id=_first_value(rec, "HUSB"),
        wife_id=_first_value(rec, "WIFE"),
        children_ids=[s.value for s in _subs(rec, "CHIL") if s.value],
        marriage=_build_event_from_tag(rec, "MARR"),
    )


def _subs(rec: Any, tag: str) -> list[Any]:
    # Use raw sub_records — ged4py's sub_tags() resolves xrefs to the linked
    # record, which destroys the xref string we need for our id-based links.
    return [s for s in rec.sub_records if s.tag == tag]


def _collect_names(rec: Any) -> list[Name]:
    names: list[Name] = []
    for sub in _subs(rec, "NAME"):
        value = sub.value
        given: str | None = None
        surname: str | None = None
        full: str
        # ged4py parses NAME into a (given, surname, suffix) tuple-like object.
        if isinstance(value, tuple) and len(value) >= 2:
            given = (value[0] or "").strip() or None
            surname = (value[1] or "").strip() or None
            full = " ".join(p for p in (given, f"/{surname}/" if surname else None) if p)
        else:
            full = str(value)
            # Try to extract /surname/ from the raw form.
            m = re.match(r"^(.*?)\s*/([^/]+)/\s*(.*)$", full)
            if m:
                given = (m.group(1) + " " + m.group(3)).strip() or None
                surname = m.group(2).strip() or None
        names.append(Name(full=full, given=given, surname=surname))
    return names


def _parse_sex(value: str | None) -> Sex:
    if value == "M":
        return Sex.MALE
    if value == "F":
        return Sex.FEMALE
    return Sex.UNKNOWN


def _build_event_from_tag(rec: Any, tag: str) -> Event | None:
    sub = next((s for s in rec.sub_records if s.tag == tag), None)
    if sub is None:
        return None
    return _build_event(sub, _EVENT_TAGS.get(tag, EventType.OTHER))


def _build_event(sub: Any, event_type: EventType) -> Event | None:
    date_sub = next((s for s in sub.sub_records if s.tag == "DATE"), None)
    place_sub = next((s for s in sub.sub_records if s.tag == "PLAC"), None)
    addr_sub = next((s for s in sub.sub_records if s.tag == "ADDR"), None)
    note_sub = next((s for s in sub.sub_records if s.tag == "NOTE"), None)
    sources = [s.value for s in sub.sub_records if s.tag == "SOUR" and s.value]

    date = _build_date(date_sub) if date_sub is not None else None
    place = _build_place(place_sub, addr_sub)

    if date is None and place is None and not sources and note_sub is None:
        # Empty event stubs (e.g. bare OCCU with no payload) carry no info.
        return None

    return Event(
        type=event_type,
        date=date,
        place=place,
        note=note_sub.value if note_sub is not None else None,
        source_ids=sources,
    )


def _build_date(date_sub: Any) -> GedcomDate | None:
    raw = _date_to_string(date_sub.value)
    if raw is None:
        return None
    year_match = _YEAR_RE.search(raw)
    year = int(year_match.group(1)) if year_match else None
    is_approx = any(raw.upper().startswith(p) for p in _APPROX_PREFIXES)
    return GedcomDate(raw=raw, year=year, is_approximate=is_approx)


def _date_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    # ged4py DateValue stringifies cleanly.
    text = str(value).strip()
    return text or None


def _build_place(place_sub: Any, addr_sub: Any) -> Place | None:
    raw = place_sub.value.strip() if place_sub is not None and place_sub.value else None
    city = state = country = None
    if addr_sub is not None:
        city = _first_value(addr_sub, "CITY")
        state = _first_value(addr_sub, "STAE")
        country = _first_value(addr_sub, "CTRY")
    if raw is None and city is None and state is None and country is None:
        return None
    return Place(raw=raw, city=city, state=state, country=country)


def _first_value(rec: Any, tag: str) -> str | None:
    sub = next((s for s in rec.sub_records if s.tag == tag), None)
    if sub is None or sub.value is None:
        return None
    value = sub.value
    if isinstance(value, str):
        return value.strip() or None
    return str(value)
