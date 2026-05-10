# Aeon-V1 Tools And Requirements Reference

Purpose: human reference only. This file is never imported or executed by Aeon.

Use this as the public, machine-neutral tool list for a fresh clone. Local app
paths, API keys, model IDs, Obsidian workspace state, transcripts, and generated
memories must stay out of Git.

## Core Tools

### Python 3.10+
- Purpose: runtime for the Aeon package, chat CLI, dashboard, runner, memory
  agents, and tests.
- Importance: required.
- Link: https://www.python.org/downloads/
- Notes: Python 3.11+ is preferred. Windows users should make sure Python is on
  PATH or use the Python launcher.

### pip
- Purpose: installs Aeon and its dependencies.
- Importance: required.
- Link: https://pip.pypa.io/en/stable/
- Notes: from the repo root, use `pip install -e .` for normal use or
  `pip install -e ".[dev]"` for development and tests.

### Git
- Purpose: clone, update, and contribute to the repository.
- Importance: required for GitHub use.
- Link: https://git-scm.com/downloads
- Notes: clone with `git clone https://github.com/jessedaustin93/aeon-v1`.

### tzdata
- Purpose: timezone support on platforms that do not ship a usable timezone
  database.
- Importance: required dependency on some systems.
- Link: https://pypi.org/project/tzdata/
- Notes: listed as a Python dependency so timestamps render consistently.

### pytest
- Purpose: runs the test suite.
- Importance: required for development.
- Link: https://docs.pytest.org/
- Notes: run `pytest` from the repository root after installing dev
  dependencies.

## Local AI Tools

### LM Studio
- Purpose: local OpenAI-compatible LLM server for chat, reflection, simulation,
  memory search planning, and optional image understanding.
- Importance: optional but recommended.
- Link: https://lmstudio.ai/
- Notes: copy `.env.lmstudio.template` to `.env`, then replace the placeholder
  model IDs with exact model IDs from your own LM Studio install.

Aeon supports separate LM Studio roles:

| Variable | Role | Typical choice |
|---|---|---|
| `AEON_V1_LLM_CHAT_MODEL` | Fast interactive chat | small/medium instruct model |
| `AEON_V1_LLM_MODEL` | General reasoning | reliable general model |
| `AEON_V1_LLM_DEEP_MODEL` | Slower reasoning/tool paths | reasoning or larger model |
| `AEON_V1_LLM_SEARCH_MODEL` | Memory search planning | concise reasoning model |
| `AEON_V1_LLM_VISION_MODEL` | Image understanding | vision-language model |

All role values are optional. If a role is blank, Aeon falls back to the general
LM Studio model where possible, then to rule-based behavior where needed.

### Future Local Providers
- Purpose: add support for other local model servers such as Ollama or llama.cpp.
- Importance: planned.
- Link: https://ollama.com/
- Notes: LM Studio is the implemented local provider today.

## Human-Facing Tools

### Aeon Dashboard And Launcher
- Purpose: browser-based local control panel for chat, image upload, runner
  controls, LM Studio launch, Obsidian launch, status, and clean shutdown.
- Importance: implemented.
- Link: not applicable.
- Notes: start with `Aeon Launcher.bat` on Windows or
  `python scripts/aeon_launcher.py` from the repo root. Machine-specific app
  launch commands belong in ignored `local/launcher_config.json`.

### Obsidian
- Purpose: human-readable graph view of the Markdown vault under `vault/`.
- Importance: optional but recommended.
- Link: https://obsidian.md/
- Notes: open the repository's `vault/` directory as an Obsidian vault. The
  local app workspace folder `vault/.obsidian/` is ignored by Git.

### Visual Studio Code Or Any Editor
- Purpose: code and Markdown editing.
- Importance: optional.
- Link: https://code.visualstudio.com/
- Notes: no specific editor is required.

## Memory And Media Tools

### Local File System
- Purpose: stores append-only JSON memories, Markdown vault notes, media files,
  chat transcripts, and runtime status.
- Importance: required.
- Link: not applicable.
- Notes: generated memories and transcripts are private runtime data and are
  ignored by Git by default. Public seed examples and documentation remain in
  the repository.

### Dashboard Image Upload
- Purpose: lets a user attach an image and ask Aeon to analyze it with a local
  vision model.
- Importance: implemented.
- Link: not applicable.
- Notes: requires an active LM Studio vision model for automatic description.
  If no vision model is available, Aeon still stores the media record and marks
  analysis unavailable.

### Audio And Video Processing
- Purpose: future ingestion of sound, speech, video frames, and richer media.
- Importance: planned.
- Link: not applicable.
- Notes: the memory architecture can store media records now; full audio/video
  analysis still needs a transcription or media-analysis layer.

## Background Operation

### Aeon Runner
- Purpose: keeps maintenance tasks alive after the launcher starts Aeon,
  including runner status, periodic vault linking, and memory-growth-triggered
  consolidation.
- Importance: implemented.
- Link: not applicable.
- Notes: run with `python scripts/aeon_runner.py` or start it from the
  dashboard. It exits cleanly when requested through the dashboard.

### Memory Consolidation
- Purpose: periodically reviews new memories for duplicate or related material
  and writes non-destructive consolidation notes.
- Importance: implemented.
- Link: not applicable.
- Notes: consolidation never erases original memories. It adds summaries and
  links so the memory set stays searchable as it grows.

## Hardware And Future Extensions

### ESP32-S3 Approval Device
- Purpose: optional hardware approval device for human-gated writes.
- Importance: experimental/planned.
- Link: see `firmware/esp32s3-auth-device/`.
- Notes: install `pip install -e ".[hardware]"` before using the hardware
  provider. The firmware source is included, but each physical device still
  needs local flashing.

### Dedicated Always-On Device
- Purpose: run Aeon continuously without tying it to a main workstation.
- Importance: planned.
- Link: https://www.raspberrypi.com/products/raspberry-pi-5/
- Notes: a small always-on machine is enough for rule-based background work.
  Local LLM inference usually needs a stronger machine than a basic Pi.

### System Service
- Purpose: supervise Aeon across reboot or crash.
- Importance: future hardening.
- Link: platform-specific.
- Notes: the current public launcher is manually started and manually stopped.
  A service wrapper can be added later.

### Docker
- Purpose: reproducible deployment with mounted `memory/` and `vault/` volumes.
- Importance: planned.
- Link: https://docs.docker.com/
- Notes: not required for current local use.

## Public Setup Rule

Anything specific to one computer must stay in a local ignored file:

- `.env`
- `local/launcher_config.json`
- `vault/.obsidian/`
- generated memory JSON files
- generated vault notes
- chat transcripts
- runtime status files
- screenshots, media uploads, and local test evidence

The repository should contain source code, tests, docs, templates, seed examples,
and generic instructions only.
