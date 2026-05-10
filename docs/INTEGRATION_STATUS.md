# Aeon-V1 Integration Status

This document summarizes what is implemented today and where future local
integrations can plug in.

## Layer Status

| Layer | Status | Module(s) | Description |
|---|---|---|---|
| Raw ingestion | Complete | `ingest.py`, `memory_store.py` | Verbatim append-only capture |
| Episodic promotion | Complete | `ingest.py`, `memory_store.py` | Promoted when importance meets threshold |
| Semantic extraction | Complete | `ingest.py`, `memory_store.py` | Promoted from concept-like inputs |
| Reflection engine | Complete | `reflect.py` | Structured analysis with rule-based safety sections |
| Task creation | Complete | `tasks.py` | Suggested tasks become stored task records |
| Decision engine | Complete | `decision.py` | Scores pending tasks and writes decision records |
| Action simulation | Complete | `simulate.py` | Proposes actions and risks without execution |
| Memory linking | Complete | `linker.py` | Adds Obsidian wikilinks between related notes |
| Search | Complete | `search.py`, `search_agent.py` | Local keyword and specialized recall paths |
| Consolidation | Complete | `consolidate.py` | Append-only consensus records for duplicate memories |
| Local LLM routing | Complete | `llm.py` | LM Studio chat, reasoning, search, vision, and tool paths |
| Embeddings | Implemented | `embeddings.py`, `consolidate.py` | Optional LM Studio embedding-assisted consolidation |
| Core protection | Complete | `memory_store.py`, `exceptions.py` | Core memory writes remain human-gated |
| Timezone handling | Complete | `time_utils.py` | UTC JSON storage with local display formatting |
| Real action execution | Out of scope | - | Simulation only by design |

## Data Flow

```text
ingest(text)
  -> store_raw()
  -> optional store_episodic()
  -> optional store_semantic()

reflect()
  -> select bounded memory sources
  -> analyze patterns, uncertainty, tasks, and confidence
  -> optionally ask LM Studio for narrative sections
  -> store_reflection()
  -> create_tasks_from_reflection()

consolidate_memories()
  -> compare episodic and semantic memories
  -> use LM Studio embeddings when available
  -> fall back to local Jaccard similarity
  -> store append-only consolidation records
```

## Local LLM Integration

Aeon is local-only today. When `AEON_V1_LLM=1`, `src/aeon_v1/llm.py` talks to
LM Studio's OpenAI-compatible local server, defaulting to:

```text
http://localhost:1234/v1
```

If LM Studio is unavailable, returns empty content, or cannot run the requested
model, LLM helpers return `None` and callers fall back to local rule-based
behavior.

JSON records that use a model include local metadata:

```json
{
  "llm_used": true,
  "llm_model": "qwen/qwen3-4b-2507",
  "llm_provider": "lmstudio"
}
```

## Intentional Boundaries

| Boundary | Reason |
|---|---|
| Local files only | No external database is required |
| LM Studio only | The current build is local-first and local-only |
| No real action execution | Action simulation stays separate from command execution |
| Core memory protection | Automated writes cannot modify `vault/core/` |
| Rule-based fallback | Aeon remains usable when local models are offline |
