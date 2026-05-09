"""Tests for append-only recursive memory consolidation."""
import json
import time

from aeon_v1 import AgentNode, Config, DataWriteAgent, MemoryStore, Orchestrator, consolidate_memories


def test_consolidation_creates_append_only_consensus(tmp_path):
    cfg = Config(tmp_path)
    cfg.consolidation_similarity_threshold = 0.6
    store = MemoryStore(cfg)

    first = store.store_semantic(
        concept="memory deduplication",
        description="Memory deduplication should preserve original source records while creating consensus.",
        tags=["memory", "dedupe"],
        importance=0.8,
        source="test",
    )
    second = store.store_semantic(
        concept="memory deduplication",
        description="Memory deduplication must preserve source records and append a consensus note.",
        tags=["memory", "dedupe"],
        importance=0.8,
        source="test",
    )

    result = consolidate_memories(cfg)

    assert result["created_count"] == 1
    record = result["created"][0]
    assert set(record["source_ids"]) == {first["id"], second["id"]}
    assert "consensus" in record["content"].lower()

    con_path = cfg.memory_path / "consolidations" / f"{record['id']}.json"
    vault_path = cfg.vault_path / "consolidations" / f"{record['id']}.md"
    assert con_path.exists()
    assert vault_path.exists()
    assert "Source memories were not deleted" in vault_path.read_text(encoding="utf-8")

    assert (cfg.memory_path / "semantic" / f"{first['id']}.json").exists()
    assert (cfg.memory_path / "semantic" / f"{second['id']}.json").exists()


def test_consolidation_is_idempotent_for_same_sources(tmp_path):
    cfg = Config(tmp_path)
    cfg.consolidation_similarity_threshold = 0.6
    store = MemoryStore(cfg)

    store.store_semantic(
        concept="duplicate lesson",
        description="Duplicate lesson records should be summarized into one consensus record.",
        tags=["lesson"],
        importance=0.7,
        source="test",
    )
    store.store_semantic(
        concept="duplicate lesson",
        description="Duplicate lesson records should be summarized into one consensus note.",
        tags=["lesson"],
        importance=0.7,
        source="test",
    )

    first = consolidate_memories(cfg)
    second = consolidate_memories(cfg)

    assert first["created_count"] == 1
    assert second["created_count"] == 0
    assert second["skipped_existing"] == 1
    assert len(list((cfg.memory_path / "consolidations").glob("*.json"))) == 1


def test_consolidator_agent_runs_through_bus(tmp_path):
    cfg = Config(tmp_path)
    cfg.consolidation_similarity_threshold = 0.6
    store = MemoryStore(cfg)
    store.store_semantic(
        concept="agent duplicate",
        description="Agent duplicate consolidation preserves sources and appends consensus.",
        tags=["agent"],
        importance=0.8,
        source="test",
    )
    store.store_semantic(
        concept="agent duplicate",
        description="Agent duplicate consolidation preserves original sources and appends consensus.",
        tags=["agent"],
        importance=0.8,
        source="test",
    )

    writer = DataWriteAgent(cfg)
    try:
        node = AgentNode(role="consolidator", config=cfg)
        result = node.run()
    finally:
        writer.close()

    assert result["role"] == "consolidator"
    assert result["created_count"] == 1
    assert result["created_ids"]


def test_orchestrator_keeps_consolidator_in_pool(tmp_path):
    cfg = Config(tmp_path)
    cfg.max_thinking_agents = 0
    cfg.consolidation_similarity_threshold = 0.6
    store = MemoryStore(cfg)
    store.store_semantic(
        concept="orchestrator duplicate",
        description="Orchestrator duplicate consolidation creates append-only consensus.",
        tags=["orchestrator"],
        importance=0.8,
        source="test",
    )
    store.store_semantic(
        concept="orchestrator duplicate",
        description="Orchestrator duplicate consolidation creates append-only consensus records.",
        tags=["orchestrator"],
        importance=0.8,
        source="test",
    )

    summary = Orchestrator(cfg).tick()

    assert summary["consolidator"]["role"] == "consolidator"
    assert summary["consolidator"]["created_count"] == 1


def test_consolidation_record_is_searchable(tmp_path):
    cfg = Config(tmp_path)
    cfg.consolidation_similarity_threshold = 0.6
    store = MemoryStore(cfg)
    store.store_semantic(
        concept="searchable consensus",
        description="Searchable consensus duplicate memories should create a durable summary.",
        tags=["searchable"],
        importance=0.8,
        source="test",
    )
    store.store_semantic(
        concept="searchable consensus",
        description="Searchable consensus duplicate memories should create a durable consensus.",
        tags=["searchable"],
        importance=0.8,
        source="test",
    )
    record = consolidate_memories(cfg)["created"][0]

    data = json.loads((cfg.memory_path / "consolidations" / f"{record['id']}.json").read_text())
    assert data["type"] == "consolidation"
    assert "searchable" in data["content"].lower()


def test_background_consolidation_runs_after_five_new_memories(tmp_path):
    cfg = Config(tmp_path)
    cfg.enable_background_consolidation = True
    cfg.consolidation_trigger_interval = 5
    cfg.consolidation_trigger_memory_types = ["semantic"]
    cfg.consolidation_similarity_threshold = 0.6
    store = MemoryStore(cfg)

    for idx in range(4):
        store.store_semantic(
            concept="background duplicate",
            description=f"Background duplicate memory should append consensus after five records. item {idx}",
            tags=["background"],
            importance=0.8,
            source="test",
        )

    time.sleep(0.2)
    early_records = [
        p for p in (cfg.memory_path / "consolidations").glob("*.json")
        if p.name != "trigger_state.json"
    ]
    assert early_records == []

    store.store_semantic(
        concept="background duplicate",
        description="Background duplicate memory should append consensus after five records. final item",
        tags=["background"],
        importance=0.8,
        source="test",
    )

    deadline = time.time() + 3
    records = []
    while time.time() < deadline:
        records = [
            p for p in (cfg.memory_path / "consolidations").glob("*.json")
            if p.name != "trigger_state.json"
        ]
        if records:
            break
        time.sleep(0.05)

    assert records, "Fifth new source memory should trigger background consolidation"
    state = json.loads((cfg.memory_path / "consolidations" / "trigger_state.json").read_text())
    assert state["last_checked_total"] == 5
