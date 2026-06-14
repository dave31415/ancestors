"""Domain models for the genealogy agent.

These Pydantic models are the public surface the agent and tools work against.
GEDCOM-library-specific types must not leak past the tool layer — anything
crossing into the agent loop should be one of the models defined here.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Sex(StrEnum):
    MALE = "M"
    FEMALE = "F"
    UNKNOWN = "U"


class EventType(StrEnum):
    BIRTH = "BIRT"
    DEATH = "DEAT"
    BURIAL = "BURI"
    MARRIAGE = "MARR"
    DIVORCE = "DIV"
    BAPTISM = "BAPM"
    IMMIGRATION = "IMMI"
    EMIGRATION = "EMIG"
    CENSUS = "CENS"
    OCCUPATION = "OCCU"
    RESIDENCE = "RESI"
    OTHER = "OTHER"


class GedcomDate(BaseModel):
    """A GEDCOM date value.

    GEDCOM dates can be exact ("10 JAN 1975"), approximate ("ABT 1797"),
    bounded ("BEF 1820", "AFT 1815"), ranged ("BET 1810 AND 1820"), or
    interpreted ("EST 1800"). We preserve the raw string and extract a
    best-guess year and an approximation flag for downstream reasoning.
    """

    raw: str
    year: int | None = None
    is_approximate: bool = False


class Place(BaseModel):
    """A place reference.

    GEDCOM exposes places via two routes: a free-form PLAC tag, and a
    structured ADDR tag with CITY/STAE/CTRY children. Geni exports here use
    ADDR heavily; PLAC shows up on older records. We capture both.
    """

    raw: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None


class Event(BaseModel):
    type: EventType
    date: GedcomDate | None = None
    place: Place | None = None
    note: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class Name(BaseModel):
    """A personal name.

    `full` preserves the GEDCOM NAME value verbatim (including the /surname/
    slashes) so a downstream reader can always recover the source form. `given`
    and `surname` are best-effort parses for matching and display.
    """

    full: str
    given: str | None = None
    surname: str | None = None


class Individual(BaseModel):
    id: str
    names: list[Name]
    sex: Sex = Sex.UNKNOWN
    birth: Event | None = None
    death: Event | None = None
    burial: Event | None = None
    other_events: list[Event] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    families_as_child: list[str] = Field(default_factory=list)
    families_as_spouse: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @property
    def primary_name(self) -> Name | None:
        return self.names[0] if self.names else None

    @property
    def display_name(self) -> str:
        name = self.primary_name
        return name.full if name else f"<unnamed {self.id}>"


class Family(BaseModel):
    id: str
    husband_id: str | None = None
    wife_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    marriage: Event | None = None
    other_events: list[Event] = Field(default_factory=list)


class GedcomDatabase(BaseModel):
    """In-memory queryable representation of a parsed GEDCOM file."""

    source_path: str
    individuals: dict[str, Individual]
    families: dict[str, Family]

    @property
    def individual_count(self) -> int:
        return len(self.individuals)

    @property
    def family_count(self) -> int:
        return len(self.families)


class IndividualQuery(BaseModel):
    """Query parameters for find_individuals.

    Every field is optional. Fields left as None are not used as match
    constraints — they do not penalize candidates. This lets the agent issue
    progressively-narrowed queries without scoring artifacts.
    """

    given: str | None = None
    surname: str | None = None
    sex: Sex | None = None
    birth_year_min: int | None = None
    birth_year_max: int | None = None
    death_year_min: int | None = None
    death_year_max: int | None = None
    place_contains: str | None = None
    limit: int = 25


class MatchScore(BaseModel):
    """Per-field scoring breakdown for an individual match.

    The agent layer is expected to reason about the breakdown, not the overall
    score in isolation. A high overall driven entirely by a name match should
    be treated differently than one supported by name + dates + place.
    """

    overall: float
    name: float | None = None
    year: float | None = None
    place: float | None = None
    sex: float | None = None


class IndividualMatch(BaseModel):
    individual: Individual
    score: MatchScore


class AncestorNode(BaseModel):
    """A node in an ancestor tree.

    `generation` is 0 for the root subject, 1 for parents, 2 for grandparents,
    and so on. `father`/`mother` are None when the parent is unknown OR when
    the generation limit was reached OR when a cycle was detected.
    """

    individual: Individual
    generation: int
    father: AncestorNode | None = None
    mother: AncestorNode | None = None


class DescendantNode(BaseModel):
    individual: Individual
    generation: int
    children: list[DescendantNode] = Field(default_factory=list)


class GapType(StrEnum):
    BIRTH_DATE_MISSING = "birth_date_missing"
    BIRTH_DATE_APPROXIMATE = "birth_date_approximate"
    BIRTH_DATE_UNSOURCED = "birth_date_unsourced"
    BIRTH_PLACE_MISSING = "birth_place_missing"
    DEATH_DATE_MISSING = "death_date_missing"
    DEATH_DATE_UNSOURCED = "death_date_unsourced"
    PARENTS_UNKNOWN = "parents_unknown"
    NO_EVENTS = "no_events"


class SortKey(StrEnum):
    """The key dimensions sort_by understands.

    Adding a new key is a deliberate act — each one needs an extractor in
    dsl.sort_by and a clear semantic (e.g. lifespan is death_year minus
    birth_year and requires both to be present).
    """

    BIRTH_YEAR = "birth_year"
    DEATH_YEAR = "death_year"
    LIFESPAN = "lifespan"
    SURNAME = "surname"
    GIVEN_NAME = "given_name"


class EvidenceGap(BaseModel):
    """A documented hole in the evidence for an individual.

    Surfacing these is the primary driver of the research agenda — the agent
    consumes evidence gaps to choose what to investigate next. Tools detect
    gaps; the agent reasons about which gaps are worth closing and how.
    """

    individual_id: str
    type: GapType
    description: str


class IdSet(BaseModel):
    """An identifier set — the algebraic flow-type for the agent DSL.

    Tools produce, refine, combine, and read IdSets. The set carries only ids
    plus optional per-id metadata (generation, score, path) attached by the
    producer that created it. The agent never inspects the underlying records
    until it explicitly hydrates via a reader tool.

    Why a wrapper rather than a bare list[str]: composition is safer when the
    flow-type is named. The validator can enforce size limits in one place,
    and producers can attach provenance the readers can later surface.
    """

    ids: list[str] = Field(default_factory=list)
    metadata: dict[str, dict[str, object]] = Field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.ids)

    def with_meta(self, id: str, key: str) -> object | None:
        return self.metadata.get(id, {}).get(key)


class ToolError(BaseModel):
    """A structured error returned to the agent in place of a result.

    Errors are values, not exceptions, on the agent surface — the dispatcher
    catches anything that escapes a tool and converts it. This keeps the
    agent loop's contract uniform: every dispatched call returns either
    `result` or `error`, never raises.
    """

    code: str
    message: str
    detail: dict[str, object] = Field(default_factory=dict)
