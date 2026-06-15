# ancestors

A genealogy research agent over a personal GEDCOM, built as a study platform
for agentic AI architecture. The genealogy corpus is the substrate; the
interesting object is the agent — its loop, its tool surface, its working
memory, its observability, and its eval harness.

This README is internal documentation: a refresher for me and a re-onboarding
crib for Claude Code when context is cold. It is not an interview artifact
and it is not aimed at outside readers.

## Running it

Three commands cover almost everything.

```bash
# Drive the agent end-to-end on one question. Streams tool calls and state
# transitions to stderr; prints the final answer to stdout.
uv run python demos/run_agent.py

# Full test suite. Hits the real GEDCOM under data/ — there are no mocks for
# the corpus layer.
uv run pytest

# Eval harness. Runs the case suites under src/ancestors/eval/cases/, prints
# a scorecard, optionally writes a new baseline.
uv run python -m ancestors.eval                  # all cases
uv run python -m ancestors.eval --suite lineage  # one suite
uv run python -m ancestors.eval --bless          # overwrite baseline.json
```

Requires `ANTHROPIC_API_KEY` in env, or a key file at
`~/.config/ancestors/keys/anthropic`. LLM calls are disk-cached by default
via `CachedAnthropic` in `src/ancestors/llm.py` — pass `--no-cache` to the
eval CLI to bypass.

## The agent loop

`src/ancestors/agent/loop.py` is an explicit state machine. Six states,
one terminal pair:

```
INTAKE     -> PLAN
PLAN       -> EXECUTE                    (LLM produces a Plan)
EXECUTE    -> EXECUTE | ASSESS | STUCK   (dispatch one queued call)
ASSESS     -> EXECUTE | PLAN |
              CHECKPOINT | ANSWER |
              STUCK                      (LLM produces an Assessment;
                                          the loop applies its updates)
CHECKPOINT -> ANSWER | STUCK             (human-gate hook; pass-through today)
ANSWER, STUCK                            (terminal)
```

Two design choices to keep in mind:

- The Planner and Assessor are `Protocol`s. The loop does not know whether
  they are LLM-backed (`LlmAgent`) or scripted (test doubles). Tests inject
  scripted ones and exercise the state machine without burning API calls.
- ASSESS is the *only* place `ResearchState` is mutated. The model returns a
  structured `Assessment` (new facts, hypotheses, dead ends, decision) and
  the loop applies those updates deterministically. There is no
  `record_fact` tool. The model never holds the pen on its own memory.

The loop has two safety valves independent of model intent: a hard call cap
and a stall detector that compares the hash of consecutive plans.

## The dispatch contract

`src/ancestors/dispatch.py` is the only execution path tools take. Three
properties matter:

1. **The agent dispatches by name.** It cannot call Python directly. A
   single `Dispatcher.dispatch(tool_name, args)` enters from the model side
   and returns a uniform `DispatchResult` envelope (`ok=True` or
   `error=ToolError`).
2. **Arguments are validated by Pydantic schemas declared per tool** in
   `src/ancestors/tool_registry.py`. Length bounds, regex on ids, enums on
   sexes/event types — these are the only thing standing between free-form
   LLM output and the tool function. They are load-bearing for
   defence-in-depth.
3. **Sets travel by opaque handle.** Tools that produce id collections
   return `set_handle = "h_N"` plus `set_size`. The model never sees raw id
   lists. The next tool composes by passing the handle in. This is what
   keeps prompts bounded as the search broadens.

The dispatcher itself has no genealogy in it. The split is:

- `dispatch.py` — generic engine: `Tool`, `Dispatcher`, `DispatchResult`,
  the hard ceilings (`MAX_INPUT_SET_SIZE`, `MAX_CALLS_PER_SESSION`, …),
  the result envelope. Any domain could reuse it.
- `tool_registry.py` — genealogy registry: per-tool argument schemas, the
  `TOOLS` dict that binds names to `dsl.py` functions, the regex for
  `PERSON_ID_RE`, the `Generations` cap, the surname-string length bound.

A new tool is a one-stop edit in `tool_registry.py` (schema + entry) plus
the function in `dsl.py`. The dispatcher does not need to change.

## Working memory

`src/ancestors/agent/state.py` holds `ResearchState`, the agent's per-question
memory. Four collections plus a history log:

- `confirmed_facts: list[Fact]` — claims believed true, with `confidence` and
  `sources`.
- `working_hypotheses: list[Hypothesis]` — claims under investigation.
- `open_questions: list[str]` — sub-questions still to resolve.
- `dead_ends: list[DeadEnd]` — paths that produced nothing, with the reason.
- `history: list[StateTransition]` — append-only log of every update.

`Fact` and `Hypothesis` both carry a `ConfidenceLevel`:
`CONFIRMED | PROBABLE | POSSIBLE | SPECULATIVE | REFUTED`. The ladder is
generic; the genealogy system prompt maps it onto the Genealogical Proof
Standard but the enum itself is not domain-specific.

All transitions return new `ResearchState` instances (`model_copy(update=…)`)
rather than mutating in place. This lets the trace replay any past state and
keeps a buggy update from corrupting prior beliefs.

`state.summary()` is the cross-turn memory. If a fact isn't rendered in that
summary, the next turn won't see it. Counts alone are not enough — the
summary renders each fact, hypothesis, and dead end as a one-liner.

## The SQLite escape hatch

`src/ancestors/sqlite_store/` builds a read-only SQLite mirror of the GEDCOM
in memory and exposes it via the `run_sql` tool. This is the back door for
queries the typed DSL can't express cleanly — recursive CTEs over the
parent_child graph, cousin-marriage detection, that sort of thing.

The connection is hardened:

- `query_only` is set, so writes are blocked.
- A progress handler enforces a wall-clock timeout (`safe_conn.py`).
- Result rows are capped at `MAX_ROWS`.
- `ATTACH DATABASE` is denied by an authorizer hook.
- Query length is capped before SQL even reaches SQLite.

Schema lives in `sqlite_store/schema.py`, the loader in `loader.py`.
`tests/test_sqlite.py` covers the three layers: schema/loader correctness,
the safety wrappers, and the `run_sql` tool surface (including the canonical
recursive-CTE ancestor walk and the cousin-marriage query).

## Observability

`src/ancestors/agent/observability.py` defines a small hook system. The
dispatcher and loop both publish events through a `HookRegistry`; observers
subscribe with no return contract — exceptions in an observer never break
dispatch.

Built-in observers:

- `make_collector()` — in-memory list, used everywhere in tests.
- `JsonlTraceWriter` — one event per line under `traces/`. This is the
  durable record. The trace viewer at `src/ancestors/trace_viewer.py` reads
  these back.

Events of interest: `before_dispatch`, `after_dispatch`, `on_error`,
`on_state_transition`, `on_plan`, `on_assessment`. The demo
(`demos/run_agent.py`) wires the trace writer and a stderr printer side by
side, so you can watch the run live and reread it later.

## The eval harness

`src/ancestors/eval/` is the regression net. The runner is corpus-agnostic;
the domain is injected through a `Corpus` protocol.

```
eval/
  corpus.py      Corpus protocol — the seam
  runner.py      Generic per-case runner (binds corpus, runs the agent)
  harness.py     run_eval(corpus, ...) — case selection, scorecard, baseline
  case.py        Case + CheckResult dataclasses
  checks.py      Check primitives (mentions_id, set_size_at_least, …)
  baseline.py    JSON load/save/diff
  reporter.py    Terminal scorecard
  cases/         Genealogy case suites: lineage / lookup / aggregation /
                 pattern / meta
  baselines/     Blessed baseline JSON lives here (currently empty)
  __main__.py    CLI — domain entry that builds GedcomCorpus and calls run_eval
```

`Corpus` is the cleanest boundary in the codebase. It exposes:
`bind()` (context manager that sets up session state), `system_prompt()`,
`tools()`, `tool_defs()`. `GedcomCorpus` in `src/ancestors/corpus.py` is the
genealogy implementation. Swapping in a non-genealogy corpus is the test of
whether the agent really is generic.

## Directory map: generic vs genealogy

The current line between agent-architecture-generic and genealogy-specific:

| Module                                | Generic? | Notes                                    |
| ------------------------------------- | -------- | ---------------------------------------- |
| `agent/loop.py`                       | generic  | State machine, no domain imports         |
| `agent/state.py`                      | generic  | ResearchState + ConfidenceLevel ladder   |
| `agent/observability.py`              | generic  | HookRegistry + JSONL trace               |
| `agent/schema.py`                     | generic  | Pydantic → Anthropic tool-defs exporter  |
| `agent/llm_agent.py`                  | mixed    | Generic loop integration; formats domain results |
| `dispatch.py`                         | generic  | Dispatcher engine, no genealogy imports  |
| `llm.py`                              | generic  | CachedAnthropic wrapper                  |
| `session.py`                          | generic  | Context-managed corpus binding           |
| `eval/` (everything except `cases/`)  | generic  | Corpus-agnostic harness                  |
| `tool_registry.py`                    | domain   | Genealogy argument schemas, TOOLS dict   |
| `dsl.py`                              | domain   | Genealogy primitives                     |
| `models.py`                           | domain   | GedcomDatabase, Individual, Family, …    |
| `tools/` (gedcom, lineage, gaps, …)   | domain   | Loader + DSL implementations             |
| `sqlite_store/`                       | domain   | GEDCOM → SQLite mirror                   |
| `genealogy_prompts.py`                | domain   | System prompt template + corpus summary  |
| `corpus.py`                           | domain   | `GedcomCorpus`                           |
| `trace_viewer.py`                     | generic  | Reads JSONL traces                       |
| `eval/cases/`                         | domain   | Genealogy case suites                    |

The seam is real but not policed by directories. Today the generic and
domain modules are siblings in one flat package. That's deliberate — see the
"no premature abstraction" note below. If a second corpus ever shows up,
that's when the package split earns its keep.

## Conventions, and what to know on resumption

For me on cold reread, and for Claude on a new session.

### Conventions

- **No premature abstraction.** This is a study platform, not a framework.
  The `Corpus` protocol exists because the eval harness needed it; nothing
  else should grow a protocol or a `core/` package "just in case." If a
  second corpus or a second agent loop ever materialises, that's the point
  at which we extract.
- **Pydantic models are the agent-facing input contract.** Argument schemas
  in `tool_registry.py` are load-bearing security boundaries, not idle
  annotations. Don't loosen a bound without thinking about what unbounded
  input does.
- **ASSESS owns ResearchState.** The model never writes facts directly —
  there is no `record_fact` tool. Adding one would break the trace replay
  story and the "model out of the bookkeeping loop" invariant.
- **Opaque handles, not raw id lists.** Tools that produce id collections
  return `set_handle`. The next tool consumes the handle. Don't add a tool
  that emits raw ids to the model.
- **Terse comments.** Comments explain *why*, not *what*. Most files have a
  short module docstring that frames the design tradeoffs; almost no
  function-level comments. Match the existing density.
- **Tests hit real data.** `tests/test_dsl.py`, `test_dispatch.py`,
  `test_sqlite.py`, `test_observability.py`, and `test_loop.py` all load the
  real GEDCOM under `data/`. There are no fixtures faking the corpus. LLM
  calls are stubbed or use `CachedAnthropic` — but tool plumbing always
  runs against the live data.

### Open threads

These are live punch-list items, not finished work:

- **Baseline still un-blessed.** The eval harness runs end-to-end but
  `eval/baselines/` is empty. Last attempted run had one FAIL and one STUCK
  case, and the `get_parents_of` traceback (below) wanted fixing before
  blessing. Resume by re-running `uv run python -m ancestors.eval`, looking
  at the scorecard, then `--bless` when you're satisfied.
- **`get_parents_of` KeyError leaks.** When the LLM passes a hallucinated
  person id, `dsl.get_parents_of` raises `KeyError` and the dispatcher
  surfaces it as `tool_exception`. It should be a structured
  `unknown_id` validation error instead — caught at the schema layer or
  early in the DSL function, returned as a typed `ToolError`. See
  `src/ancestors/dsl.py:101` for the raise site.
- **Anthropic prompt caching is off.** No `cache_control` markers anywhere
  in `agent/llm_agent.py` or `llm.py`. The system prompt + tool defs are
  resent uncached on every call. Turning it on is a few lines and should
  cut wall time on long conversations significantly.
- **Retry-with-cache-bypass.** When `CachedAnthropic` returns a cached but
  malformed structured response, we have no mechanism to detect that and
  retry with the cache bypassed. Worth wiring up when prompt caching lands.

### Where to look first

If you're picking this up cold and need the highest-leverage files:

- `src/ancestors/agent/loop.py` — the state machine. The header docstring
  describes the transition table and the two safety valves.
- `src/ancestors/dispatch.py` — the security boundary. Header docstring
  lays out the five design properties.
- `src/ancestors/tool_registry.py` — the tool surface. Read the schemas
  before the `TOOLS` dict.
- `src/ancestors/agent/state.py` — the working-memory model.
- `src/ancestors/eval/corpus.py` and `src/ancestors/corpus.py` — the
  generic/domain seam, both ends.
