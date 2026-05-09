# Scripts

`scripts/` contains human-facing entry points and utility commands. These scripts are thin wrappers around the package in `src/aeon_v1/`.

Run scripts from the repo root unless a command says otherwise.

## Launchers

| Script | Purpose |
|---|---|
| `aeon_chat.py` | Opens the terminal chat interface. |
| `aeon_launcher.py` | Starts the local browser dashboard. |
| `aeon_runner.py` | Starts the always-on background runner. Usually launched by the dashboard. |

Windows helper files in the repo root:

```text
Aeon Chat.bat
Aeon Launcher.bat
Aeon Launcher.vbs
```

The `.vbs` launcher hides the extra terminal window and opens the dashboard flow.

## Memory Utilities

| Script | Purpose |
|---|---|
| `ingest_text.py` | Ingests direct text, a file, or stdin into memory. |
| `search_memory.py` | Searches local memory from the command line. |
| `run_reflection.py` | Runs one reflection pass. |

Examples:

```bash
python scripts/ingest_text.py "Important project goal: ship Aeon-V1."
python scripts/ingest_text.py --file notes.txt --source journal
python scripts/search_memory.py "recursive memory"
python scripts/run_reflection.py
```

## Task And Simulation Utilities

| Script | Purpose |
|---|---|
| `manage_tasks.py` | Lists tasks, selects decisions, runs simulation loops, and manages task flow. |

Useful commands:

```bash
python scripts/manage_tasks.py tasks
python scripts/manage_tasks.py decide
python scripts/manage_tasks.py simulate
python scripts/manage_tasks.py loop
```

## Test And Diagnostic Utilities

| Script | Purpose |
|---|---|
| `run_live_test.py` | Runs a local live behavior check. |
| `test_tool_calling.py` | Exercises LLM tool-calling paths. |

## Notes For Operators

- Scripts should not contain machine-specific paths.
- Machine-specific app launch commands belong in `local/launcher_config.json`.
- If a script writes memory, it should go through package APIs such as `ingest()` or approved write paths.
- If a script is only for local experiments, keep generated outputs out of Git unless they are generalized examples.
