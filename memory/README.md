# Memory Store

`memory/` is Aeon's machine-readable storage. Records here are JSON files used by the program. Human-readable mirrors live in `vault/`.

This directory is local state, not just documentation. Treat it carefully.

## Main Rule

Raw memory is append-only and verbatim. Derived layers can summarize, connect, or consolidate, but they should not erase or rewrite the source records.

## Memory Layers

| Directory | Purpose |
|---|---|
| `raw/` | Exact stored inputs. This is the source of truth for what was actually written. |
| `episodic/` | Event-like summaries derived from important raw records. |
| `semantic/` | Concept/rule-like records derived from important inputs. |
| `reflections/` | Append-only synthesis over episodic and semantic memory. |
| `consolidations/` | Duplicate/overlap summaries. These preserve source IDs and do not delete originals. |
| `media/` | Image/media records, source paths, and optional vision-model descriptions. |
| `chat/` | JSONL chat transcripts used to restore recent dashboard/terminal context. |

## Task And Agent State

| Directory | Purpose |
|---|---|
| `tasks/` | Tasks created from reflections or operator input. |
| `decisions/` | Decision records for selected tasks. |
| `simulations/` | Simulation plans. These do not execute real commands. |
| `evaluations/` | Evaluations of simulation outcomes. |
| `agents/` | Agent node records. |
| `logs/` | Audit and activity logs. |

## Governance And Tooling State

| Directory | Purpose |
|---|---|
| `staging/` | Proposed writes waiting for validation/approval. |
| `approved/` | Approved write records. |
| `schemas/` | Tool/schema manifests and related definitions. |
| `tool_additions/` | Proposed or stored tool additions. |
| `runtime/` | Dashboard/runner status, stop files, and runtime heartbeat data. |

## How Search Uses Memory

`search.py` reads JSON memory records and vault Markdown notes. `SearchAgent` uses that search backend for explicit memory questions such as:

```text
what do you remember about Obsidian?
search memories for dashboard
```

Those direct memory-search answers should return `llm_used=false` and source memory IDs.

## What Not To Commit

Be careful with real personal memory. Generalized examples are fine; private local memories should stay local.

Before pushing to GitHub, check for:

- private conversations,
- local machine paths,
- API keys or secrets,
- private Obsidian workspace files,
- test artifacts that expose personal data.
