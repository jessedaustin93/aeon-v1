# Vault

`vault/` is the human-readable Markdown view of Aeon's memory. It is designed to open cleanly in Obsidian, but the files are normal Markdown and can be read with any editor.

The machine-readable source records live in `memory/`.

## What The Vault Is For

- Reviewing memories as notes instead of JSON.
- Browsing wikilinks and graph connections in Obsidian.
- Keeping operator-facing core notes separate from generated memory.
- Making Aeon's memory auditable by a human.

## Main Sections

| Directory | Purpose |
|---|---|
| `raw/` | Markdown mirrors of exact raw memory records. |
| `episodic/` | Event-like memory notes. |
| `semantic/` | Concept and rule notes. |
| `reflections/` | Reflection notes over memory. |
| `consolidations/` | Duplicate/overlap consolidation notes. |
| `media/` | Media memory notes and analyzed descriptions. |
| `tasks/` | Task notes. |
| `decisions/` | Decision notes. |
| `simulations/` | Simulation notes. |
| `evaluations/` | Evaluation notes. |
| `agents/` | Agent and tool-related notes. |
| `core/` | Human-controlled core memory. Automated code must not modify this area. |

## Obsidian

Install Obsidian locally and open this directory as a vault:

```text
vault/
```

No plugins are required. Generated notes use YAML frontmatter and wikilinks like:

```text
[[raw/abc12345|readable-title]]
```

The ignored `.obsidian/` directory stores local Obsidian UI/workspace state. Do not rely on it for portable project behavior.

## Core Vault Rule

`vault/core/` is human-gated. Aeon can read it as context, but automated ingestion, linking, reflection, and consolidation should not write into it.

Use core notes for stable identity/rules that a human intentionally chooses to preserve.

## Relationship To Memory

The vault mirrors useful records from `memory/`; it is not a replacement for JSON. When debugging behavior:

1. Check `memory/` for the machine source record.
2. Check `vault/` for the human-readable note.
3. Run linking if relationships are stale.

## Privacy

Vault notes may contain real personal memory. Do not push private local vault content unless it has been generalized and intentionally reviewed.
