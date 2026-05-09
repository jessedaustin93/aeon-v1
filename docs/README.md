# Documentation

`docs/` holds the human-readable design and setup documents for Aeon-V1. The root README is the overview; this directory has the deeper material.

## Documents

| File | Purpose |
|---|---|
| `setup_from_github.md` | Fresh-clone setup instructions for Python, editable install, LM Studio, Obsidian, launcher, tests, and optional hardware. |
| `architecture.md` | High-level system architecture and layer layout. |
| `memory_model.md` | Detailed explanation of raw, episodic, semantic, reflection, consolidation, media, and vault memory. |
| `recursive_learning_loop.md` | How ingestion, reflection, tasks, simulation, evaluation, and consolidation form a loop. |
| `tools_manifest.md` | Human reference for required, optional, planned, and experimental tools. |
| `obsidian.md` | How the local Markdown vault works with Obsidian. |
| `auth_device.md` | Hardware approval-device notes. |
| `mempalace.md` | Memory palace concept notes. |
| `INTEGRATION_STATUS.md` | Snapshot of integration progress and open edges. |

## How To Use These Docs

Start with:

1. `setup_from_github.md` when installing or helping another person run Aeon.
2. `architecture.md` when trying to understand the whole system.
3. `memory_model.md` when debugging recall or storage behavior.
4. `recursive_learning_loop.md` when working on reflection, consolidation, or background learning.
5. `tools_manifest.md` when checking what outside tools are required or optional.

## Documentation Rules

- Keep the root README concise.
- Put subsystem details in the closest README or doc file.
- Do not document private local memory as public setup information.
- Do not include machine-specific paths except as examples clearly marked local.
- If behavior changes in code, update the matching doc in the same work session.
