# aeon_v1 Package

This directory contains Aeon-V1's core implementation. The modules are grouped by behavior below so you can understand the system without reading every file first.

## Human Interfaces

| Module | Purpose |
|---|---|
| `chat_cli.py` | Terminal chat interface. Stores chat turns, retrieves memory, routes self/search questions, calls the LLM path when needed, and logs transcripts. |
| `dashboard.py` | Browser dashboard and local control surface. Handles chat, image upload, status cards, start/stop controls, and clean shutdown. |
| `launcher_config.py` | Loads portable launcher defaults and ignored local overrides. |
| `runtime.py` | Reads/writes runtime status files and memory counts for dashboard/runner status. |
| `runner.py` | Always-on background loop for memory maintenance and status heartbeat. |

## Memory Storage

| Module | Purpose |
|---|---|
| `config.py` | Central paths and runtime settings. Loads `.env` and defines memory/vault locations. |
| `memory_store.py` | Writes raw, episodic, semantic, reflection, task, decision, simulation, evaluation, and related JSON/Markdown records. |
| `ingest.py` | Stores raw text, scores importance, and promotes important records to episodic/semantic memory. |
| `media.py` | Stores uploaded images and optional vision-model descriptions as media memory. |
| `linker.py` | Adds related-memory wikilinks to vault Markdown notes while avoiding protected core files. |
| `search.py` | Keyword/token search over JSON memory and vault notes. This is the current search backend. |

## Memory Readers

| Module | Purpose |
|---|---|
| `search_agent.py` | Natural-language memory search. Handles questions like "what do you remember about X?" and returns stored records with `llm_used=false`. |
| `self_inspection_agent.py` | Read-only repo self-inspection. Handles questions like "what are you?" and "what can you do?" by reading README/docs/source-map files. |
| `memory_index_agent.py` | Tool-call bridge for LLM memory queries during deeper reasoning paths. |

## Learning And Synthesis

| Module | Purpose |
|---|---|
| `reflect.py` | Builds reflection records from episodic/semantic memory. |
| `consolidate.py` | Finds likely duplicates/overlaps and writes append-only consolidation records without erasing originals. |
| `background_consolidation.py` | Count-based trigger helpers for running consolidation after memory growth. |
| `tasks.py` | Creates and persists tasks suggested by reflections. |
| `decision.py` | Selects next tasks based on priority/confidence signals. |
| `simulate.py` | Creates simulation plans for proposed actions. It does not execute real commands. |
| `evaluate.py` | Records simulation evaluations and turns them into episodic learning. |

## Agents And Governance

| Module | Purpose |
|---|---|
| `agent.py` | Generic agent records and role behavior. |
| `orchestrator.py` | Coordinates agent ticks, task flow, and maintenance passes. |
| `data_write_agent.py` | Detects and stages proposed memory/tool writes. |
| `write_agent.py` | Applies approved write proposals. |
| `approval_agent.py` | Human approval gate. |
| `schemas.py` | Validates proposal/message shapes and allowed memory types/actions. |
| `security.py` | Path guard and safety checks for file operations. |
| `write_guard.py` | Runtime write authorization context. |
| `manifest_agent.py` | Tracks tool/schema manifest drift. |
| `hardware_auth_provider.py` | Optional ESP32-S3 approval device integration. |

## Tools And Calls

| Module | Purpose |
|---|---|
| `tools.py` | Tool definition store. |
| `builtin_tools.py` | Built-in tool metadata. |
| `tool_calls.py` | Persistent records for proposed/approved tool calls. |
| `bus.py` | In-process request/reply bus used by tool-call and memory-index paths. |

## LLM Integration

| Module | Purpose |
|---|---|
| `llm.py` | Optional local LM Studio adapter. Handles chat messages, tool calling, model role routing, and vision calls. |

Aeon must still run when LLM calls fail. LLM helpers return `None` on provider errors so callers can fall back to local behavior.

Model roles are intentionally separated:

```text
llm_chat_model    normal conversational voice
llm_deep_model    slower reasoning/tool-call path
llm_search_model  memory-search planner, often Mistral/Ministral
llm_vision_model  image understanding
```

`SearchAgent` can use `llm_search_model` to plan better search terms, but chat replies still use `llm_chat_model`.

## Time And Errors

| Module | Purpose |
|---|---|
| `time_utils.py` | UTC/local timestamp helpers. |
| `exceptions.py` | Shared exception classes. |
| `__init__.py` | Public package exports. |

## Main Answer Paths

Chat and dashboard questions are routed in this order:

1. Image upload path if an image is attached.
2. `SelfInspectionAgent` for self/system questions.
3. `SearchAgent` for explicit memory-search questions.
4. Normal LLM chat with retrieved context when enabled.
5. Local fallback response when no LLM answer is available.

This separation is intentional. It lets operators tell whether an answer came from source files, stored memory, or an LLM.

## Safety Rules In Code

- Raw memory is preserved verbatim.
- Derived memory layers append records; they do not replace raw memory.
- Consolidation appends synthesis and does not erase sources.
- Core vault files are protected from automated writes.
- Agent writes must pass validation and approval.
- Simulations do not execute real commands.
- Machine-specific launcher configuration belongs in ignored `local/launcher_config.json`.
