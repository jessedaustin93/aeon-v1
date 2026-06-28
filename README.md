# Aeon-V1

Aeon-V1 is a local-first AI memory, chat, and learning system. It stores what happens in plain local files, reflects on those records, exposes a local dashboard, can use LM Studio models when available, and keeps the operator in control of any write-governed behavior.

The short version: Aeon is not a single text file and it is not a cloud chatbot wrapper. It is a local program made of memory stores, agents, a dashboard, optional local LLM routing, Obsidian-readable notes, and tests that keep the pieces honest.

## What Is In This Repo

| Area | What it does | Where to read more |
|---|---|---|
| Source package | Core Python modules for memory, chat, agents, dashboard, LLM routing, media, governance, and self-inspection | [src/aeon_v1/README.md](src/aeon_v1/README.md) |
| Scripts | Command-line and launcher entry points | [scripts/README.md](scripts/README.md) |
| Memory store | Machine-readable JSON memory records and runtime state | [memory/README.md](memory/README.md) |
| Obsidian vault | Human-readable Markdown view of memory | [vault/README.md](vault/README.md) |
| Documentation | Architecture, setup, tools, Obsidian, recursive loop, and memory model docs | [docs/README.md](docs/README.md) |
| Tests | Regression tests for the implemented system | [tests/README.md](tests/README.md) |
| Local overrides | Ignored machine-specific launcher configuration | [local/README.md](local/README.md) |
| Firmware | Optional ESP32-S3 hardware approval device | [firmware/esp32s3-auth-device/README.md](firmware/esp32s3-auth-device/README.md) |

## Current Capabilities

Aeon-V1 currently supports:

- Local chat through terminal and browser dashboard.
- Raw, episodic, semantic, reflection, consolidation, and media memory layers.
- Natural-language memory search for questions like "what do you remember about X?"
- Read-only self-inspection for questions like "what can you do?" and "how do you work?"
- Image upload and image memory through an optional LM Studio vision model.
- Optional local LM Studio model routing for chat, general reasoning, deep reasoning, vision, and embeddings.
- Obsidian vault output with Markdown notes and wikilinks.
- An always-on local runner for background linking and memory-growth consolidation.
- Append-only duplicate consolidation that writes summaries without erasing source records.
- Task, decision, simulation, evaluation, and governed write pipelines.
- Manifest drift monitoring for tool/schema changes.
- Optional ESP32-S3 hardware approval support.
- A pytest suite covering the system behavior.

Planned or future work includes richer audio/video ingestion, stronger service supervision, and packaged deployment.

## How The Main Pieces Fit

```text
Human chat / dashboard
        |
        v
chat_cli.py / dashboard.py
        |
        +--> SelfInspectionAgent reads repo docs/source map for "what are you?" questions
        +--> SearchAgent reads stored memories for "what do you remember?" questions
        +--> LLM adapter uses LM Studio when enabled
        |
        v
MemoryStore writes JSON in memory/
        |
        +--> raw records preserve exact inputs
        +--> episodic/semantic records derive useful summaries and concepts
        +--> media records store analyzed image descriptions
        +--> reflections/consolidations add append-only synthesis
        |
        v
vault/ mirrors useful records as Markdown for Obsidian
```

The dashboard and runner sit beside this flow. The dashboard is the local control surface. The runner keeps background maintenance alive after you manually start Aeon.

## Quick Start

For a fresh GitHub download, read [docs/setup_from_github.md](docs/setup_from_github.md). That file has the full setup path for Python, editable install, LM Studio, Obsidian, and optional hardware.

Basic install:

```bash
git clone https://github.com/jessedaustin93/aeon-v1
cd aeon-v1
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Open terminal chat:

```bash
python scripts/aeon_chat.py
```

Open the local dashboard:

```bash
python scripts/aeon_launcher.py
```

On Windows, you can also use:

```text
Aeon Chat.bat
Aeon Launcher.bat
```

The dashboard defaults to:

```text
http://127.0.0.1:8765
```

## Local LLM Setup

Aeon can run without an LLM. When an LLM is enabled, it is an optional reasoning and response layer, not the memory store itself.

Recommended local setup:

1. Install LM Studio.
2. Load the models you want to use.
3. Copy `.env.lmstudio.template` to `.env`.
4. Set the model role variables in `.env`.
5. Start the LM Studio local server.
6. Launch Aeon.

Important model roles:

```text
AEON_V1_LLM_MODEL          general fallback model
AEON_V1_LLM_CHAT_MODEL     normal chat model
AEON_V1_LLM_DEEP_MODEL     deeper reasoning/tool-call model
AEON_V1_LLM_SEARCH_MODEL   memory-search planner model
AEON_V1_LLM_MUSIC_MODEL    song-pipeline planning model
AEON_V1_LLM_VISION_MODEL   image understanding model
AEON_V1_LLM_BASE_URL       usually http://localhost:1234/v1
AEON_V1_LLM_MUSIC_BASE_URL optional remote machine-local endpoint for the music role
```

Recommended local split:

```text
Chat:   Qwen chat/instruct model
Deep:   Qwen thinking/reasoning model
Search: Mistral/Ministral-style model for planning memory queries
Vision: Qwen VL or another vision-capable model
Music: small instruct model on the machine that owns the music pipeline
```

The search model does not become Aeon's chat voice. It only helps `SearchAgent` turn fuzzy recall questions into concrete memory search terms.
The music model also remains internal: explicit library-management requests route
to it, while the operator continues to interact with one Aeon identity.

See [docs/setup_from_github.md](docs/setup_from_github.md) and [docs/tools_manifest.md](docs/tools_manifest.md) for more.

## Memory And Truth Modes

Aeon has multiple answer paths. This matters when checking whether it is remembering or generating.

| User asks | Path used | What to look for |
|---|---|---|
| "What do you remember about X?" | `SearchAgent` over `memory/` | `llm_used=false`, memory IDs from stored records |
| "What are you?" or "What can you do?" | `SelfInspectionAgent` over repo docs/source map | `llm_used=false`, source IDs like `self:README.md` |
| Normal conversation | LLM when enabled, with retrieved context | `llm_used=true` unless fallback |
| Image upload | `media.py` plus optional vision model | media record in `memory/media/` |

The clearest proof of real memory is a deterministic canary:

1. Store a strange exact phrase.
2. Ask Aeon to remember that exact phrase.
3. Confirm `llm_used=false`.
4. Confirm the reply includes the memory ID and exact stored content.

## Safety Model

Aeon is local-first and file-backed.

- Raw memory is append-only and preserved verbatim.
- Derived memories do not replace raw records.
- Reflection and consolidation append new records instead of deleting old ones.
- Core vault files are human-gated.
- Agent-proposed writes go through validation and approval paths.
- Real command execution is out of scope for Aeon-V1.
- Machine-specific launcher paths stay in ignored local config.

## Directory Map

```text
README.md                     system overview
docs/                         human documentation
scripts/                      runnable entry points
src/aeon_v1/                  core package
memory/                       machine-readable records and runtime state
vault/                        Markdown/Obsidian view of memory
local/                        ignored local launcher overrides
tests/                        pytest suite
firmware/esp32s3-auth-device/ optional hardware approval device
```

Each major directory has its own README so operators can understand that section without reading every source file.

## Good First Files To Read

Start here:

1. [docs/setup_from_github.md](docs/setup_from_github.md) - install and run from a fresh clone.
2. [src/aeon_v1/README.md](src/aeon_v1/README.md) - how the Python modules fit together.
3. [memory/README.md](memory/README.md) - what each memory layer means.
4. [vault/README.md](vault/README.md) - how Obsidian fits in.
5. [scripts/README.md](scripts/README.md) - how to launch and operate Aeon.

## Development

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run focused tests while working:

```bash
pytest tests/test_chat_cli.py tests/test_search_agent.py tests/test_self_inspection_agent.py -q
```

Keep private local memory and machine-specific app paths out of Git unless they are intentionally generalized examples.
