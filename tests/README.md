# Tests

`tests/` contains the pytest suite for Aeon-V1.

The tests are not just code checks. They document system guarantees: append-only memory, protected core vault behavior, LLM fallback behavior, dashboard behavior, search behavior, self-inspection, media ingestion, consolidation, and governance.

## Run Tests

From the repo root:

```bash
pytest
```

Focused examples:

```bash
pytest tests/test_chat_cli.py -q
pytest tests/test_search_agent.py tests/test_self_inspection_agent.py -q
pytest tests/test_launcher_dashboard.py -q
```

## Important Test Areas

| Test file | What it covers |
|---|---|
| `test_memory_store.py` | Raw memory, markdown mirrors, titles, search, reflections, core protection. |
| `test_chat_cli.py` | Chat prompt building, transcripts, retrieval context, fallback behavior. |
| `test_search_agent.py` | Natural-language memory search and anti-echo behavior. |
| `test_self_inspection_agent.py` | Repo/doc self-inspection answers. |
| `test_launcher_dashboard.py` | Dashboard, runner, launcher config, image chat, transcript restore. |
| `test_media.py` | Image/media memory ingestion and search. |
| `test_consolidation.py` | Duplicate/overlap consolidation behavior. |
| `test_lmstudio_parallel.py` | LM Studio concurrency and model-role routing. |
| `test_llm.py` | LLM fallback and prompt helpers. |
| `test_agent_orchestrator.py` | Agent node and orchestrator behavior. |
| `test_manifest_agent.py` | Tool/schema manifest tracking. |

## Testing Rules

- Prefer focused tests for new behavior.
- Do not require an external LLM for normal test success.
- Mock LM Studio or vision calls unless a test is explicitly a local live test.
- Use temporary directories for memory/vault writes.
- Keep tests deterministic and local-first.
