# Setup From GitHub

This checklist describes what is ready immediately after cloning Aeon-V1 and
what still needs local setup.

Aeon-V1 is local-first. The repository includes the source code, docs, tests,
seed memory structure, Obsidian-compatible vault structure, LM Studio template,
and ESP32-S3 firmware source. Local secrets, model selections, installed Python
packages, running LM Studio servers, and flashed hardware are intentionally not
stored in GitHub.

## 1. Clone The Repository

```bash
git clone https://github.com/jessedaustin93/aeon-v1
cd aeon-v1
```

## 2. Install Aeon-V1

For normal local use:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install anthropic        # Anthropic provider support
pip install -e ".[hardware]" # ESP32-S3 USB approval provider support
```

## 3. Run The Test Suite

```bash
pytest
```

Expected result for the current suite:

```text
612 passed, 1 skipped
```

## 4. Run Aeon Without An LLM

Aeon works without a model. In this mode, ingestion, search, reflection,
task creation, decision selection, simulation, and the chat shell use local
rule-based behavior.

Launch the chat interface:

```bash
python scripts/aeon_chat.py
```

Or, after editable install:

```bash
aeon-chat
```

Windows users can also launch chat directly:

```text
Aeon Chat.bat
```

Or open the local dashboard/launcher:

```text
Aeon Launcher.bat
```

From a terminal:

```bash
python scripts/aeon_launcher.py
```

The dashboard opens at `http://127.0.0.1:8765` by default and can start the Aeon runner, launch LM Studio, launch Obsidian, display status, and request clean shutdown.

## 5. Enable LM Studio

Start from the generic template:

```bash
cp .env.lmstudio.template .env
```

PowerShell:

```powershell
Copy-Item .env.lmstudio.template .env
```

Then edit `.env` and replace these placeholders with exact model IDs from
LM Studio:

```env
AEON_V1_LLM_CHAT_MODEL=your-fast-chat-model-id
AEON_V1_LLM_MODEL=your-general-model-id
AEON_V1_LLM_DEEP_MODEL=your-deep-reasoning-model-id
AEON_V1_LLM_SEARCH_MODEL=your-memory-search-model-id
AEON_V1_LLM_VISION_MODEL=your-vision-model-id
```

You may use one model for every role, or split them by purpose:

| Variable | Purpose |
|---|---|
| `AEON_V1_LLM_CHAT_MODEL` | Fast interactive chat responses |
| `AEON_V1_LLM_MODEL` | General reflection/simulation reasoning |
| `AEON_V1_LLM_DEEP_MODEL` | Deeper tool-calling reflection/simulation paths |
| `AEON_V1_LLM_SEARCH_MODEL` | Memory-search planning before the chat model answers |
| `AEON_V1_LLM_VISION_MODEL` | Optional image understanding for dashboard uploads |

Good generic starting point:

- use a fast instruct/chat model for chat
- use a reasoning-capable model for deep work
- use a small reasoning-capable model for memory search
- use a vision-language model only when image upload/analysis is needed

Leave any role blank if you do not want to use it yet. Aeon will fall back to
the general model where possible, then to local rule-based behavior if the model
or server is unavailable.

In LM Studio, start the OpenAI-compatible local server. The default Aeon URL is:

```text
http://localhost:1234/v1
```

If LM Studio uses another URL, update:

```env
AEON_V1_LLM_BASE_URL=http://localhost:1234/v1
```

Dashboard image uploads work best when `AEON_V1_LLM_VISION_MODEL` is set to an active vision-capable LM Studio model. If the model is missing, unloaded, or fails, Aeon still stores the image and marks analysis unavailable for later reprocessing.

## 6. Optional Tool Calling

Tool calling lets reflection and simulation ask Aeon's `MemoryIndexAgent` for
bounded local memory context instead of receiving all memories inlined in the
prompt.

Enable it in `.env`:

```env
AEON_V1_LLM_TOOL_CALLING=1
```

Not every LM Studio model supports OpenAI-style tool calling well. If a
tool-calling request fails or returns no final content, Aeon retries the deep
model without tools and then falls back to rule-based behavior if needed.

## 7. Optional Anthropic Provider

Install the optional package:

```bash
pip install anthropic
```

Set environment variables:

```bash
export AEON_V1_LLM=1
export AEON_V1_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key_here
```

PowerShell:

```powershell
$env:AEON_V1_LLM="1"
$env:AEON_V1_LLM_PROVIDER="anthropic"
$env:ANTHROPIC_API_KEY="your_key_here"
```

## 8. Optional Obsidian Vault

Install Obsidian locally from the official download page:

```text
https://obsidian.md/download
```

Windows users may also be able to install with:

```powershell
winget install Obsidian.Obsidian
```

Then open the repository's `vault/` directory as an Obsidian vault:

```text
aeon-v1/vault/
```

Start from `index.md`. Aeon-generated notes use YAML frontmatter and `[[wikilinks]]`, and `link_memories()` can refresh related-memory links.

See `docs/obsidian.md` for the full local workflow.

## 9. Optional Launcher Configuration

The launcher works without a local config file, but exact app paths vary by
machine. To customize local startup commands without committing private paths:

```bash
cp local/launcher_config.example.json local/launcher_config.json
```

PowerShell:

```powershell
Copy-Item local/launcher_config.example.json local/launcher_config.json
```

Then edit `local/launcher_config.json`. You can set:

- dashboard host/port
- runner poll interval
- LM Studio command or base URL
- Obsidian command, vault name, or vault path

`local/launcher_config.json` is ignored by Git.

## 10. Optional ESP32-S3 Approval Device

The repository includes firmware source for the hardware approval token:

```text
firmware/esp32s3-auth-device/
```

Install the hardware extra:

```bash
pip install -e ".[hardware]"
```

Flash the firmware with PlatformIO:

```bash
cd firmware/esp32s3-auth-device
pio run -t upload
pio device monitor
```

See `docs/auth_device.md` for protocol details.

## 11. What Is Ready Immediately

- Python package source under `src/aeon_v1/`
- CLI scripts under `scripts/`
- Windows launcher `Aeon Chat.bat`
- Windows launcher/control panel `Aeon Launcher.bat`
- local JSON memory directories
- Obsidian-compatible Markdown vault directories
- dashboard image upload and media-memory directories
- test suite
- LM Studio template `.env.lmstudio.template`
- ESP32-S3 firmware source

## 12. What Must Stay Local

- `.env`
- API keys
- exact local model selections
- LM Studio server state
- Obsidian app/workspace state in `vault/.obsidian/`
- launcher overrides in `local/launcher_config.json`
- runner status files in `memory/runtime/`
- generated runtime memories and transcripts
- flashed hardware state

These are intentionally not committed to GitHub.
