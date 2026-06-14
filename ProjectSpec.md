# Genealogy Research Agent — CLAUDE.md

## Project Purpose

This project is a multi-step reasoning agent that performs genealogical research. It is
explicitly designed as a study system for agentic AI architecture, with genealogy as the
domain. The goal is to build something that exercises long-horizon planning, memory, tool
use, external model inference, and robust recovery from failure — the core skills of
production agentic systems.

The domain is not incidental. The researcher has deep expertise in genealogy, including
active research into Irish Donegal ancestry (the Gallagher line, Peter Gallagher b. ~1798,
Carndonagh, County Donegal, emigrated to New Brunswick 1821) and a DAR/SAR lineage through
Dr. Lancelot Johnston (Continental Congress surgeon, 1777). Real GEDCOM data from this
research is available and should be used. This means the agent will be evaluated by someone
who knows when it is right and when it is confabulating.

---

## Architecture Overview

The system has three distinct layers. Keep them cleanly separated.

### Layer 1: Tool Layer (deterministic, no LLM)

Python functions that do exactly one thing and return structured Pydantic objects. No
reasoning here. No LLM calls here. These are the hands of the system.

### Layer 2: Agent Loop (LLM reasoning)

The core reasoning loop. Receives a research question or task, maintains a research state,
decides which tools to call, interprets results, updates beliefs, detects when it has enough
information to answer, and decides when to ask for human input. This is where LLM reasoning
is justified — at the boundary of ambiguity, conflicting evidence, and incomplete
information.

### Layer 3: Interface Layer

CLI for now. Accepts a research question from the user, runs the agent loop, streams
reasoning steps to the terminal, and presents a final answer or report. Human-in-the-loop
checkpoints should be explicit and documented.

---

## The Central Design Principle

**LLM reasoning should only be used where deterministic code is genuinely insufficient.**

Before adding any LLM reasoning step, ask: could a rule, a query, or an optimizer handle
this? If yes, use that instead. The LLM layer earns its place at:

- Reconciling conflicting evidence from multiple sources
- Assessing whether a record is a plausible match for a known individual
- Generating hypotheses about missing links
- Deciding whether to keep searching or report uncertainty
- Producing narrative explanations of evidence chains
- Interpreting ambiguous or incomplete records in context

The LLM layer does NOT do:
- GEDCOM parsing (that is a tool)
- Date arithmetic (that is a tool)
- Record retrieval (that is a tool)
- Structured data formatting (that is a tool)

---

## Tool Specifications

All tools return Pydantic models. All tools raise typed exceptions on failure. The agent
loop handles exceptions — tools do not swallow errors silently.

### GEDCOM Tools

```python
load_gedcom(path: str) -> GedcomDatabase
    # Parses a GEDCOM file into an in-memory queryable structure.
    # Returns individuals, families, events, sources, notes.

get_individual(db: GedcomDatabase, id: str) -> Individual
    # Returns a single individual by GEDCOM ID with all associated events,
    # family links, and source citations.

find_individuals(db: GedcomDatabase, query: IndividualQuery) -> list[Individual]
    # Query by name, birth year range, birthplace, death year range, etc.
    # Returns ranked list of matches with match confidence scores.

get_family(db: GedcomDatabase, id: str) -> Family
    # Returns a family unit with spouse and child links resolved.

get_ancestors(db: GedcomDatabase, id: str, generations: int) -> AncestorTree
    # Returns an ancestor tree up to N generations.

get_descendants(db: GedcomDatabase, id: str, generations: int) -> DescendantTree

find_evidence_gaps(db: GedcomDatabase, id: str) -> list[EvidenceGap]
    # Returns a list of undocumented claims: birth dates without sources,
    # parent links without documentation, etc. This is a key tool for
    # driving the research agenda.
```

### Research State Tools

The research state is the agent's working memory for a session. It persists across tool
calls within a session and should be serializable to disk for resumption.

```python
ResearchState:
    question: str                          # Original research question
    confirmed_facts: list[Fact]            # Facts with source citations, high confidence
    working_hypotheses: list[Hypothesis]   # Plausible claims under investigation
    open_questions: list[str]              # Unresolved sub-questions
    searched_sources: list[SearchRecord]   # What has been searched, when, what was found
    confidence_assessments: dict[str, float]  # Per-claim confidence 0.0-1.0
    dead_ends: list[DeadEnd]               # Searches that returned nothing useful

update_research_state(state: ResearchState, update: StateUpdate) -> ResearchState
    # Immutable update — returns new state. All updates are logged.

get_research_summary(state: ResearchState) -> str
    # Returns a human-readable summary of current research status.
    # Used for human-in-the-loop checkpoints.
```

### External Source Tools

These tools are stubs initially. Build the interface first; real API integration comes
later. Each stub should return realistic synthetic data so the agent loop can be developed
and tested independently.

```python
search_familysearch(query: PersonQuery) -> list[ExternalRecord]
    # Searches FamilySearch.org. Initially a stub returning synthetic records.

search_ancestry(query: PersonQuery) -> list[ExternalRecord]
    # Searches Ancestry.com. Initially a stub.

search_findagrave(query: PersonQuery) -> list[ExternalRecord]
    # Searches Find a Grave. Initially a stub.

fetch_record(source: ExternalRecord) -> RecordDetail
    # Fetches full detail for a record identified in a search result.
```

### Analysis Tools

```python
assess_record_match(individual: Individual, record: ExternalRecord) -> MatchAssessment
    # Deterministic scoring of how well a record matches a known individual.
    # Checks: name similarity, date consistency, place plausibility, family
    # composition match. Returns a structured score with per-field breakdown.
    # NOTE: This tool scores. The agent reasons about the score.

find_conflicts(facts: list[Fact]) -> list[Conflict]
    # Identifies conflicting claims about the same individual or event.
    # Returns structured conflict objects with the conflicting sources identified.

compute_date_range(events: list[Event]) -> DateConstraints
    # Given a set of life events, computes implied birth/death year ranges.
    # E.g., if someone appears in an 1850 census as age 35, born ~1815.
```

### Report Tools

```python
generate_proof_summary(
    individual: Individual,
    evidence_chain: list[Fact],
    state: ResearchState
) -> ProofSummary
    # Produces a structured proof summary suitable for DAR/SAR applications.
    # Follows GPS (Genealogical Proof Standard) structure.
    # The LLM writes the narrative; this tool formats and validates the structure.

generate_research_report(state: ResearchState) -> ResearchReport
    # Produces a full research report: what was found, what is uncertain,
    # what remains to be investigated, and recommended next steps.
```

---

## Agent Loop Design

The agent loop is the core of the system. It should be implemented as an explicit state
machine, not an emergent behavior. The states are:

```
INTAKE        → Parse the research question. Identify what kind of question it is.
               Types: lookup, proof, discovery, gap_analysis, conflict_resolution.

PLAN          → Given the question type and current research state, decide what to
               do next. This is an LLM reasoning step. Output is a typed plan object
               with an ordered list of intended tool calls and their justifications.

EXECUTE       → Call the next tool in the plan. Handle exceptions. Update research state.

ASSESS        → After each tool call, assess: does the result change the plan?
               Is there a conflict with existing beliefs? Is confidence sufficient
               to answer? This is an LLM reasoning step.

CHECKPOINT    → If the agent is about to take an irreversible action (e.g., write a
               report, assert a high-confidence claim), pause and show the human the
               current state and intended next step. Wait for confirmation.

ANSWER        → Synthesize the research state into a final answer or report.
               Explicitly state confidence level and what would change the conclusion.

STUCK         → If the agent cannot make progress (all leads exhausted, conflicting
               evidence unresolvable), report this explicitly with the current state.
               Do not confabulate. Do not assert claims without evidence.
```

The transition from ASSESS back to PLAN is the core loop. The agent should be able to
run for many iterations before reaching ANSWER or STUCK.

---

## Reasoning Patterns the Agent Should Use

These are explicit patterns the LLM should be prompted to follow. Document them here so
the prompts can be evaluated against the spec.

### Source Evaluation
Before accepting any external record, assess: Who created this record? When? For what
purpose? What are the known error rates for this record type? A death certificate birth
date is less reliable than a birth certificate birth date. An index is less reliable than
an image of the original. A transcription may contain errors the original does not.

### Triangulation
A single source is a claim. Two independent sources agreeing is evidence. Three independent
sources agreeing is strong evidence. The agent should explicitly track source independence
— two sources that both derive from the same original document are not independent.

### Reasonably Exhaustive Search
Before concluding a record doesn't exist, the agent should be able to articulate what
sources were searched, what search strategies were used, and why the absence of a record
is meaningful given those searches. "I didn't find it" is not the same as "it doesn't
exist."

### Confidence Calibration
Every claim in the research state has a confidence score. The agent should be able to
explain each score. Claims should not exceed the evidence. "Probably" and "possibly" are
meaningful distinctions. High confidence should require multiple independent sources.

### Explicit Uncertainty
When the evidence is insufficient, say so. Do not fill gaps with plausible stories. The
output of the STUCK state is valuable information, not a failure.

---

## Key Domain Facts

These are known facts about the primary research subjects. Use these to validate agent
behavior — a well-functioning agent should be able to recover these from the GEDCOM file
and reason about them correctly.

- **Peter Gallagher**: born ~1798, Carndonagh area, County Donegal, Ireland. Emigrated to
  New Brunswick, Canada, 1821. The emigration date and origin are working hypotheses with
  partial evidence, not confirmed facts. The agent should treat them accordingly.

- **Dr. Lancelot Johnston**: Continental Congress surgeon, 1777. The lineage through him
  is the basis for a DAR/SAR application. The agent should be able to identify and assess
  the evidence chain for this lineage.

- The researcher has an H-index of 65 and a PhD in Physics — precision and evidence
  standards matter. Vague or unsupported claims will be noticed immediately.

---

## What Good Output Looks Like

A well-functioning agent produces outputs that a human genealogist would recognize as
rigorous. Specifically:

- Every factual claim is sourced or explicitly flagged as unsourced
- Confidence levels are stated and justified
- Conflicting evidence is surfaced, not suppressed
- The reasoning trace is readable and follows GPS principles
- The agent knows what it doesn't know and says so
- Reports distinguish between "confirmed," "probable," "possible," and "speculative"

A poorly functioning agent:
- States facts without sources
- Ignores conflicts between records
- Fills gaps with plausible-sounding but unsupported claims
- Reports high confidence without sufficient evidence
- Stops searching too early or keeps searching past the point of diminishing returns

---

## Development Sequence

Build in this order. Each stage should be independently testable before moving to the next.

**Stage 1: Tool layer and data model**
Build and test all tools against the real GEDCOM file. Verify that the data model captures
all necessary information. Do not touch the agent loop yet.

**Stage 2: Research state**
Build the ResearchState model and its update/summary tools. Write unit tests that verify
state transitions are correct and logged. This is the agent's memory — it must be reliable.

**Stage 3: Agent loop skeleton**
Implement the state machine with stub LLM calls (return canned responses). Verify that
the control flow, exception handling, and checkpoint logic work correctly before adding
real LLM reasoning.

**Stage 4: LLM reasoning integration**
Replace stub LLM calls with real calls. Start with the ASSESS step — it's the most
contained and easiest to evaluate. Then PLAN. Use the Anthropic API directly (claude-sonnet-4-6),
not a framework. Write the prompts explicitly and version-control them alongside the code.

**Stage 5: Stub external sources**
Build the stub external source tools with realistic synthetic data. Test the full agent
loop on questions that require external source consultation.

**Stage 6: Evaluation**
Before building real external source integration, build an evaluation harness. Define a
set of research questions with known answers (derivable from the GEDCOM file). Measure
whether the agent reaches the correct answer, the correct confidence level, and the correct
identification of gaps. This is the observability and evaluation methodology the job
description calls out explicitly.

**Stage 7: Real external source integration**
Only after the agent loop is validated on known questions. Add real FamilySearch and
Ancestry API integration. Evaluate on questions that require external sources.

---

## Implementation Notes

- **Language**: Python 3.11+
- **LLM**: Anthropic API, claude-sonnet-4-6. Call the API directly. No LangChain, no
  LangGraph. The agent loop is explicit code, not framework magic.
- **Structured outputs**: Pydantic v2 throughout. Every tool input and output is a typed
  Pydantic model. LLM outputs that need to be structured should use tool/function calling
  or JSON mode, not regex parsing of free text.
- **Logging**: Every tool call, every LLM call, every state transition should be logged
  with timestamp, inputs, outputs, and latency. This is the observability layer. Build it
  from day one, not as an afterthought.
- **Testing**: pytest. Each tool has unit tests. The agent loop has integration tests
  against the stub external sources. Evaluation questions are a separate test suite.
- **GEDCOM library**: Use `python-gedcom` or `gedcompy` for parsing. Do not write a GEDCOM
  parser from scratch.
- **No autonomous writes**: The agent never writes to the GEDCOM file without explicit
  human confirmation at a CHECKPOINT. The GEDCOM file is the source of truth and must not
  be corrupted by agent errors.

---

## The Deeper Purpose

This system is a study platform for agentic AI architecture. Every design decision should
be made consciously and documented. When you make a choice — to use a tool rather than LLM
reasoning, to add a checkpoint, to represent state in a particular way — write a comment
explaining why. The goal is not just a working system but a system whose architecture is
legible and defensible.

The questions this system is designed to answer through building:

1. Where exactly is the boundary between deterministic code and LLM reasoning in a
   production agentic system?
2. How do you design a memory system that is reliable enough to trust across a long
   research session?
3. How do you build observability into an agent loop so you can evaluate and improve it?
4. How do you handle the human-in-the-loop pattern without making it so disruptive that
   the agent becomes useless?
5. What does a good evaluation harness for an agentic system look like?

These are the questions a Staff MLE at Boston Dynamics would be expected to have thought
through carefully.
