# Aeon-V1 Memory Model

## Design Principles

1. **Raw memory is sacred.** Once written, a raw record is never modified, summarized over, or deleted. The `text` field is immutable from the moment `store_raw()` returns.
2. **Promotion is additive and append-only.** Creating an episodic or semantic note never removes or replaces its source. Reflections always create new files — they never overwrite prior reflections.
3. **IDs are permanent; titles are human-readable.** Every record has a stable 8-character UUID prefix (`id`) that never changes, and a derived `title` field that is short and readable. Filenames are `{id}.md` so they always resolve correctly.
4. **Every layer is human-readable.** All memory exists as plain Markdown alongside the JSON.
5. **Links are explicit and readable.** Wikilinks use `[[subdir/id|Readable Title]]` format so they are navigable in Obsidian and legible in any text editor.
6. **Core memory requires human approval.** The `vault/core/` directory is never written to by any automated process.

---

## ID vs Title

| Field | Purpose | Stable? |
|---|---|---|
| `id` | 8-char UUID prefix — the permanent reference | Yes — never changes |
| `title` | Short hyphen-joined words — for human readability | No — derived from content |

The `id` is the canonical identifier used in filenames (`{id}.md`), `raw_ref` links, and `source_ids` lists.  
The `title` is used as Obsidian wikilink display text: `[[raw/a1b2c3d4|remember-this-important]]`.

---

## Memory Types

### Raw Memory

The exact, verbatim capture of an input.

- **Created by:** `ingest()`
- **Stored at:** `memory/raw/{id}.json`, `vault/raw/{id}.md`
- **Schema:** `memory/schemas/raw_memory.schema.json`
- **Key field:** `text` — immutable after creation. No code path ever rewrites this file.
- **Title:** First 6 alphanumeric words of the text, lowercased and hyphen-joined.

Raw memories are analogous to sensory memory in cognitive models: everything is stored before anything is evaluated.

### Episodic Memory

A summarized, meaningful event derived from one raw memory.

- **Created by:** `ingest()` when `importance >= config.importance_threshold` (default 0.5)
- **Stored at:** `memory/episodic/{id}.json`, `vault/episodic/{id}.md`
- **Schema:** `memory/schemas/episodic_memory.schema.json`
- **Key fields:** `summary`, `raw_ref` (ID of source raw memory)
- **Source link:** `[[raw/{raw_id}|{raw_title}]]` — readable link back to the raw source

### Semantic Memory

A reusable concept, rule, pattern, or lesson that transcends a single event.

- **Created by:** `ingest()` when episodic promotion occurs AND semantic keywords are detected
- **Stored at:** `memory/semantic/{id}.json`, `vault/semantic/{id}.md`
- **Schema:** `memory/schemas/semantic_memory.schema.json`
- **Key fields:** `concept`, `description`
- **Title:** The concept name, slugified.

### Core Memory

Long-term stable identity, rules, and goals. **Manually curated only.**

- **Stored at:** `vault/core/` (Markdown only — no automated JSON writing)
- **Written by:** Humans, after reviewing reflection suggestions
- **Protected by:** `vault/core/PROTECTED.md` — explains the protection contract
- **Automated processes:** May never write here. Reflections write suggestions into `vault/reflections/` only.
- **Enforcement:** `_write_markdown` raises `CoreMemoryProtectedError` if any code attempts to write a file inside `vault/core/` while `config.allow_core_modification` is `False`. `linker.py` silently skips `vault/core/` files in the same condition. This is enforced in code, not just documented.

See `vault/core/PROTECTED.md` for the human-readable protection contract.

### Reflections

Recursive analysis of what has been stored. Always appended, never replaced.

- **Created by:** `reflect()`
- **Stored at:** `memory/reflections/{id}.json`, `vault/reflections/{id}.md`
- **Schema:** `memory/schemas/reflection.schema.json`
- **Key fields:** `content` (Markdown analysis), `source_ids` (IDs of reviewed memories)
- **Title:** `reflection-YYYYMMDD-HHMM` (date-stamped)
- **Append-only:** Every `reflect()` call creates a new file. Prior reflections are never modified.
- **Safety limits:** Only reviews episodic and semantic memories (not prior reflections, unless `config.allow_reflection_on_reflections = True`).

### Consolidations

Append-only consensus records for likely duplicate or overlapping memories.

- **Created by:** `consolidate_memories()`, the `consolidator` agent during orchestrator ticks, or background count-driven maintenance after new memories are written
- **Stored at:** `memory/consolidations/{id}.json`, `vault/consolidations/{id}.md`
- **Key fields:** `content`, `source_ids`, `source_types`, `links`
- **Purpose:** Preserve source memories while adding a human-readable consensus note.
- **Safety contract:** Consolidation never deletes, rewrites, or merges source records. It only appends a new record that links back to the originals.
- **Default background trigger:** every 5 new raw/episodic/semantic memories while an Aeon process is alive (`Config.consolidation_trigger_interval`).

### Media

Append-only records for uploaded images and, later, video/audio inputs.

- **Created by:** dashboard image upload or `ingest_image_file()` / `ingest_image_bytes()`
- **Stored at:** `memory/media/{id}.json`, `vault/media/{id}.md`
- **Original files:** copied under `memory/media/uploads/`
- **Key fields:** `media_type`, `source_path`, `original_name`, `description`, `analysis_status`, `analysis_model`
- **Vision model:** optional LM Studio model from `Config.llm_vision_model` / `AEON_V1_LLM_VISION_MODEL`
- **Safety contract:** The uploaded file is preserved. If analysis fails, Aeon records `analysis_status: unavailable` rather than discarding the file.

### Tasks and Agents

Human-maintained operational notes.

- `vault/tasks/` — implied future actions derived from reflection review
- `vault/agents/` — per-agent configuration notes

---

## Obsidian Link Format

All links use the `[[subdir/id|Readable Title]]` format:

```markdown
[[raw/a1b2c3d4|remember-this-important-concept]]
[[episodic/e5f6a7b8|i-learned-that-building]]
[[semantic/c9d0e1f2|memory-layering]]
```

- `subdir/id` is the vault-relative path to `vault/{subdir}/{id}.md` — always resolves correctly.
- `Readable Title` is the `title` field from the memory record.
- Links in the `links:` frontmatter list and in note body sections use this same format.

---

## Frontmatter Schema

Every vault note includes YAML frontmatter:

```yaml
---
id: a1b2c3d4
title: remember-this-important-concept
type: raw
created: 2024-01-15T10:30:00
source: manual
importance: 0.75
tags:
  - learning
  - project
links:
  - "[[episodic/e5f6a7b8|i-learned-that-building]]"
---
```

---

## Importance Scoring

Weighted heuristic — replace `memory_store.py:_score_importance` with an LLM call when ready.

**Base score:** 0.10  
**Additive signals (first match wins for multi-word phrases):**

| Signal | Weight |
|---|---|
| "i learned" | +0.20 |
| "key insight" | +0.20 |
| "must not forget" | +0.20 |
| "critical" | +0.15 |
| "important" | +0.15 |
| "remember" | +0.12 |
| "project" | +0.12 |
| "goal" | +0.12 |
| "discovered" / "realized" | +0.10 each |
| "need to" | +0.08 |
| "should " / "will " / "always" / "never" | +0.05 each |

**Length bonus:** +0.05 for ≥100 chars, +0.10 for ≥300 chars  
**Maximum:** 1.0 (clamped)  
**Default promotion threshold:** 0.5 (configurable via `Config.importance_threshold`)

### Why the base is 0.10

Plain sentences with no signal words score 0.10–0.15.  
Any meaningful statement with one or two keywords scores 0.25–0.35.  
The threshold at 0.5 requires multiple clear signals, ensuring only genuinely notable input is promoted.

---

## Reflection Safety Limits

Configured in `Config`:

| Field | Default | Effect |
|---|---|---|
| `max_memories_per_reflection` | 50 | Caps the number of source memories per pass (most recent kept) |
| `allow_reflection_on_reflections` | `False` | Prevents reflections from recursively reviewing prior reflections |
| `allow_core_modification` | `False` | Documents intent; not yet enforced in code |

---

## Tag Extraction

Tags are keyword-mapped from the raw text:

| Keyword in text | Tag applied |
|---|---|
| project | project |
| goal | goal |
| learned | learning |
| important | important |
| remember | recall |
| error | error |
| bug | bug |
| idea | idea |
| question | question |
| task | task |
| experiment | experiment |
| concept | concept |
| pattern | pattern |
| rule | rule |
