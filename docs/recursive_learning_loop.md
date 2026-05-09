# Recursive Learning Loop

## Concept

Aeon-V1 implements a three-phase loop that mirrors how biological memory consolidation works:

```
 ┌──────────────────────────────────────────────────┐
 │                                                  │
 │   INPUT ──► RAW ──► EPISODIC ──► SEMANTIC        │
 │                                        │         │
 │                    REFLECTION ◄────────┘         │
 │                        │                         │
 │                        └──► TASKS / CORE         │
 │                                  │               │
 │                                  └──► (next INPUT)
 └──────────────────────────────────────────────────┘
```

Each cycle through the loop increases the depth and quality of stored knowledge without losing the originals.

---

## Phase 1: Ingestion

**Trigger:** A call to `ingest(text)` or the `scripts/ingest_text.py` script.

**What happens:**

1. Text is received and stored verbatim as a **raw memory**.
2. An importance score is computed from keyword heuristics.
3. Tags are extracted from the content.
4. If importance >= threshold (default 0.5):
   - An **episodic memory** is created with a one-line summary.
   - If semantic keywords are detected:
     - A **semantic memory** is created with a named concept and description.
5. All memories are written to both `memory/` (JSON) and `vault/` (Markdown) simultaneously.

**Design note:** Raw storage is unconditional. Even low-importance inputs are preserved because their significance may not be apparent until a future reflection pass.

---

## Phase 2: Reflection

**Trigger:** A call to `reflect()` or the `scripts/run_reflection.py` script.

**What happens:**

1. All episodic and semantic memories are loaded (reflections excluded by default).
2. Safety limits are applied: source cap, duplicate guard, low-value guard.
3. Structured analysis is computed via `_analyse()`:
   - Tag frequencies reveal dominant themes and repeated patterns.
   - Uncertainty signals are detected across source text.
   - Task phrases are extracted and listed as suggested actions.
   - Core memory candidates are identified (suggestions only — never written automatically).
   - A confidence score [0.0–1.0] is computed from source count, tag diversity, and uncertainty.
4. A reflection note is generated with 7 required sections:
   - **What Was Learned** — high-importance memories with readable wikilinks
   - **Important Memories Reviewed** — all source memories with importance scores
   - **New Patterns Noticed** — repeated tags and recurring themes
   - **Conflicts or Uncertainty** — uncertainty signals found in source text
   - **Suggested Tasks** — task phrases extracted from memories
   - **Suggested Core Memory Updates** — human-review-required suggestions only
   - **Reflection Quality** — confidence score and analysis notes
5. The reflection note and its metadata are stored in `memory/reflections/` and `vault/reflections/`.

**Duplicate guard:** If `config.skip_duplicate_reflections = True` (default), a pass whose source IDs exactly match a prior reflection is skipped — no redundant notes accumulate.

**Low-value guard:** If the source count falls below `config.min_reflection_sources` (default: 1), the pass is skipped unless `config.allow_low_value_reflections = True`.

**Reflection JSON metadata fields** (Layers 2 + 4, stored alongside core fields):

| Field | Type | Description |
|---|---|---|
| `confidence` | float [0, 1] | Computed from source count, tag diversity, uncertainty |
| `source_types` | dict | `{"episodic": N, "semantic": M}` |
| `suggested_tasks` | list[str] | Extracted task phrases from source memories |
| `suggested_core_updates` | list[str] | Core memory candidates (suggestions only) |
| `detected_patterns` | list[str] | Repeated tags and high-importance clusters |
| `uncertainty_notes` | list[str] | Memories containing uncertainty-signal words |
| `generated_at` | ISO timestamp | When the analysis was computed |
| `llm_used` | bool | Whether an LLM enhanced the narrative sections |
| `llm_model` | str or null | Model name used, or null |
| `llm_provider` | str or null | Provider name, or null |

**LLM integration (Layer 4):** When `config.llm_enabled` is True and `ANTHROPIC_API_KEY` is set, `_generate_reflection` calls `generate_text()` from `llm.py` to enhance sections 1, 3, 4, and 5 with LLM-synthesized narrative. Sections 2, 6, and 7 are always rule-based for safety (no LLM invention of source memories, no bypassing of the core memory warning). The system falls back silently to fully rule-based if LLM is unavailable.

---

## Phase 2b: Consolidation

**Trigger:** Every 5 new source memories by default, a call to `consolidate_memories()`, or an orchestrator tick through the persistent `consolidator` agent.

**What happens:**

1. Episodic and semantic memories are loaded.
2. Likely duplicate or strongly overlapping memories are grouped with word-overlap plus shared-tag signals.
3. Groups that already have a consolidation record are skipped.
4. New groups produce append-only consensus records in `memory/consolidations/` and `vault/consolidations/`.
5. Source records are linked but never deleted, rewritten, or merged.

Consolidation is recursive memory housekeeping: it lets Aeon compress repeated ideas into a readable consensus layer while preserving the original evidence trail.

The background trigger is count-driven rather than clock-driven. Each raw, episodic, or semantic memory write updates a persisted local counter, and the fifth new source memory starts a daemon-thread consolidation pass. This means consolidation follows memory growth instead of waiting for CLI input or a timer.

---

## Phase 3: Decision and Action Simulation (Layer 3)

**Trigger:** A call to `select_next_task()` or `python scripts/manage_tasks.py decide`.

### 3a — Task Creation (automatic, triggered by reflection)

When `reflect()` completes, it automatically converts its `suggested_tasks` list into
structured task objects stored in `memory/tasks/` and `vault/tasks/`.

- Near-duplicate tasks are blocked (Jaccard word-overlap >= `duplicate_task_similarity_threshold`, default 0.8).
- Tasks above `max_pending_tasks` (default 100) are also blocked.
- Every task carries: `id`, `title`, `description`, `source_reflection_id`, `created_at`, `status`, `priority`, `confidence`, `tags`, `links`.
- Status lifecycle: `pending` → `selected` → `simulated` → `completed` (last step is human-gated).

### 3b — Decision Engine

`select_next_task()` in `decision.py`:

1. Loads all `pending` tasks.
2. Scores each by `priority × 0.5 + confidence × 0.3`.
3. Selects the highest scorer, marks it `selected`.
4. Writes an append-only decision record to `memory/decisions/` and `vault/decisions/`.

Decision records include: `selected_task_id`, `reason`, `confidence`, `alternatives_considered`, `source_links`.

### 3c — Action Simulation

`simulate_action(task)` in `simulate.py`:

**Safety guarantee:** No subprocess, os.system, network, or execution primitive is imported or called.
All output is local files only. `enable_real_actions` is always `False` — there is no execution path to enable it.

1. If `config.llm_enabled` is True: calls `generate_text()` via `llm.py` to produce `proposed_action`, `expected_outcome`, and risk bullets.
2. Falls back to rule-based `_propose_action()` / `_expected_outcome()` / `_estimate_risks()` for any field the LLM did not return.
3. Flags destructive/external/network signals with explicit warnings (both LLM and rule-based paths).
4. Writes a simulation record to `memory/simulations/` and `vault/simulations/`.
5. Marks the task `simulated`.

Simulation records include: `task_id`, `proposed_action`, `expected_outcome`, `risks`, `required_human_approval` (always True by default), `source_links`, `llm_used`, `llm_model`, `llm_provider`.

**Why no real execution?** Aeon-V1 is a memory and reasoning framework. Execution capability requires explicit human approval for each action, proper sandboxing, and rollback mechanisms — none of which exist yet. This layer is the planning substrate that prepares for future tool use without risking uncontrolled system changes.

### Layer 3 CLI

```bash
# List pending tasks
python scripts/manage_tasks.py tasks

# Select the best pending task and write a decision record
python scripts/manage_tasks.py decide

# Simulate the most recently selected task
python scripts/manage_tasks.py simulate

# decide + simulate in one call
python scripts/manage_tasks.py loop
```

### Layer 3 Config Fields

| Field | Default | Effect |
|---|---|---|
| `enable_real_actions` | `False` | Safety lock — no execution path exists to enable |
| `max_pending_tasks` | `100` | Blocks new task creation above this count |
| `duplicate_task_similarity_threshold` | `0.8` | Jaccard threshold for near-duplicate blocking |
| `require_human_approval_for_simulation` | `True` | Flags every simulation for human review |

---

## Phase 4: Integration (human-gated)

**Trigger:** Human review of decision and simulation records in `vault/decisions/` and `vault/simulations/`.

**What happens:**

1. A human reads the simulation record and decides whether to act on it.
2. Key insights are manually promoted to `vault/core/` as long-term core memory.
3. Completed tasks are marked `completed` manually.
4. Actions taken feed back into new ingestions, closing the recursive loop.

This is the recursive element: reflections → tasks → decisions → simulations → human action → new inputs → richer memories.

---

## Reflection and Consolidation Intervals

The `Config.reflection_interval` field (default: 10) is a placeholder for future automation.
A future version could automatically trigger a reflection pass every N ingestions, or on a schedule.
For now, reflections are triggered manually.

Consolidation already has a count-driven trigger:

- `Config.enable_background_consolidation`: enabled by default outside tests.
- `Config.consolidation_trigger_interval`: default `5`.
- `Config.consolidation_trigger_memory_types`: default `["raw", "episodic", "semantic"]`.

---

## Future Phases

| Future Phase | Description |
|---|---|
| **Real Execution** | Sandboxed tool-use with explicit human approval gates per action |
| **Forgetting** | Archive low-importance memories that haven't been linked in N days |
| **Consolidation** | Merge duplicate semantic memories into a single stronger concept |
| **Dreaming** | Offline reflection pass that cross-links memories without new input |
| **Agent Memory** | Per-agent episodic stores that feed into a shared semantic layer |
| **Distillation** | Generate a compressed "worldview" summary for LLM system prompt injection |

---

## Example Cycle (Layers 1–3)

```
Day 1:
  ingest("I learned that recursive functions need a base case. This is a key concept.")
  → raw/a1b2c3d4.json
  → episodic/e5f6a7b8.json
  → semantic/c9d0e1f2.json

Day 2:
  ingest("Important project milestone: all unit tests passing.")
  → raw/b2c3d4e5.json
  → episodic/f6a7b8c9.json

  reflect()
  → reflections/r7b8c9d0.json    (7-section structured note, confidence=0.42)
  consolidate_memories()
  → consolidations/c1d2e3f4.json  (append-only consensus over overlapping memories)
  → tasks/t1a2b3c4.json          (auto-created: "review test coverage")
  → tasks/t5d6e7f8.json          (auto-created: "document passing criteria")

  decide()
  → decisions/d9a0b1c2.json      (selected t1a2b3c4, priority=0.5, confidence=0.42)

  simulate()
  → simulations/s3d4e5f6.json    (proposed_action: "Review and act on: review test coverage...")
                                  (required_human_approval: True)

Human reviews vault/simulations/s3d4e5f6.md:
  → Takes real action manually
  → Ingests result: ingest("Reviewed test coverage — found 3 uncovered paths.")
  → Loop continues with richer memories
```
