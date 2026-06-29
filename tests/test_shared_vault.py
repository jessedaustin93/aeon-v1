from aeon_v1 import Config
from aeon_v1.chat_cli import build_response, retrieve_context
from aeon_v1.shared_vault import load_startup_context, search_shared_vault


def _config(tmp_path):
    cfg = Config(tmp_path / "aeon")
    cfg.master_vault_path = tmp_path / "Master-Vault"
    cfg.master_vault_enabled = True
    return cfg


def test_shared_vault_search_is_source_labeled_and_read_only(tmp_path):
    cfg = _config(tmp_path)
    note = cfg.master_vault_path / "Projects" / "Aeon" / "README.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Aeon\nThe shared handoff token is cobalt-seven.\n", encoding="utf-8")
    before = sorted(str(path) for path in cfg.master_vault_path.rglob("*"))

    results = search_shared_vault("cobalt-seven", cfg)

    after = sorted(str(path) for path in cfg.master_vault_path.rglob("*"))
    assert results[0]["match_type"] == "master-vault"
    assert results[0]["memory"]["path"] == "Projects/Aeon/README.md"
    assert results[0]["memory"]["source"] == "master-vault"
    assert before == after


def test_shared_vault_is_optional(tmp_path):
    cfg = Config(tmp_path)
    assert search_shared_vault("anything", cfg) == []
    assert load_startup_context(cfg) == ""


def test_startup_context_uses_operating_order(tmp_path):
    cfg = _config(tmp_path)
    for relative in ("AI/SharedMemory.md", "AI/Workflow.md", "AI/AEON.md"):
        path = cfg.master_vault_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.stem}\n", encoding="utf-8")
    context = load_startup_context(cfg)
    assert context.index("AI/SharedMemory.md") < context.index("AI/Workflow.md")
    assert context.index("AI/Workflow.md") < context.index("AI/AEON.md")


def test_chat_context_keeps_shared_results_distinct(tmp_path):
    cfg = _config(tmp_path)
    note = cfg.master_vault_path / "Projects" / "Widget.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Widget\nShared-only fact: amber protocol.\n", encoding="utf-8")
    results = retrieve_context("amber protocol", cfg, limit=5)
    assert any(result["match_type"] == "master-vault" for result in results)
    assert not (cfg.memory_path / "raw").exists()


def test_chat_prompt_receives_master_vault_startup_context(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.llm_enabled = True
    cfg.llm_tool_calling = False
    note = cfg.master_vault_path / "AI" / "SharedMemory.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Shared Memory\nCross-assistant rule: preserve provenance.\n", encoding="utf-8")
    captured = {}

    def fake_generate_chat(messages, config):
        captured["messages"] = messages
        return "ack"

    monkeypatch.setattr("aeon_v1.chat_cli.generate_chat", fake_generate_chat)
    response = build_response("hello", [], [], cfg, index_agent=None)

    assert response == "ack"
    assert "Cross-assistant rule: preserve provenance" in captured["messages"][1]["content"]
