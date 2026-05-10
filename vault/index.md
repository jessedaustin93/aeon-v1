---
title: Aeon-V1 Memory Vault
type: index
created: 2026-04-29T00:00:00+00:00
---

# Aeon-V1 Memory Vault

Welcome to the Aeon-V1 knowledge graph.

Open this `vault/` folder in Obsidian to explore graph view, backlinks, and tag search. Obsidian is optional; all files are plain Markdown readable in any editor.

Start here: [[Aeon Memory Dashboard]]

---

## Memory Layers

| Layer | Link | Purpose |
|---|---|---|
| Raw | [[Raw Memory]] | Verbatim captures, never modified after ingestion |
| Episodic | [[Episodic Memory]] | Summarized meaningful events derived from raw |
| Semantic | [[Semantic Memory]] | Reusable concepts, rules, and patterns |
| Reflections | [[Reflections]] | Recursive analysis of episodic and semantic memory |
| Tasks | [[Tasks]] | Structured task objects derived from reflection |
| Decisions | [[Decisions]] | Append-only decision records from the selection engine |
| Simulations | [[Simulations]] | Proposed-action records, never executed automatically |
| Core | [[Core Memory]] | Stable identity, goals, and rules, human-gated |

---

## How To Navigate

- Graph view: use Obsidian's graph view to see how notes connect.
- Tag search: filter memories by topic tag.
- Backlinks: trace any insight back to its source.
- Wikilinks: derived memory records link toward their sources when available.

---

## Memory Flow

```text
Raw (verbatim, immutable)
  -> Episodic (summarized event, if importance >= threshold)
      -> Semantic (reusable concept, if semantic keyword detected)

Episodic + Semantic
  -> Reflection (recursive analysis, structured 7-section note)
      -> Tasks (suggested actions converted to task objects)
          -> Decision (best task selected by scoring engine)
              -> Simulation (proposed-action record, human review required)

Core Memory (human-gated, never written automatically)
```

---

## Example Notes

- [[_generated/raw/example_raw|example_raw]]: an example raw memory input
- [[_generated/episodic/example_episodic|example_episodic]]: the episodic note derived from the example raw
- [[_generated/semantic/example_semantic|example_semantic]]: the semantic concept extracted from the example
- [[_generated/reflections/example_reflection|example_reflection]]: the reflection note produced after analysis

---

[[Aeon Memory Dashboard]] | [[Core Memory]] | [[Reflections]] | [[Tasks]] | [[Decisions]] | [[Simulations]]
