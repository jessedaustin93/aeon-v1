"""Tests for Layer 4 — optional LLM reasoning integration.

Coverage:
- Config defaults and AEON_V1_LLM env toggle
- generate_text returns None when disabled
- generate_text returns None when LM Studio is unavailable
- Reflection fallback: full rule-based path works without LLM
- Simulation fallback: full rule-based path works without LLM
- Mocked LLM output is correctly inserted into reflection Markdown
- Mocked LLM output is correctly inserted into simulation record
- llm_used/llm_model/llm_provider metadata is accurate in both cases
- No subprocess or execution primitives in llm.py
- All 7 Markdown sections always present regardless of LLM state
- Prompt builders produce non-empty prompts containing safety language
"""
import ast
import os
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from aeon_v1 import Config, generate_text, ingest, reflect, simulate_action
from aeon_v1.llm import (
    build_reflection_prompt,
    build_simulation_prompt,
    generate_image_description,
    _detect_lmstudio_vision_model,
    parse_reflection_sections,
    parse_simulation_sections,
    resolve_lmstudio_vision_model,
)
from aeon_v1.tasks import TaskStore


SRC_DIR = Path(__file__).parent.parent / "src" / "aeon_v1"

# ---------------------------------------------------------------------------
# Config — LLM fields and env toggle
# ---------------------------------------------------------------------------

def test_llm_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AEON_V1_LLM", raising=False)
    cfg = Config()
    assert cfg.llm_enabled is False


def test_llm_enabled_via_env(monkeypatch):
    monkeypatch.setenv("AEON_V1_LLM", "1")
    cfg = Config()
    assert cfg.llm_enabled is True


def test_llm_env_zero_means_disabled(monkeypatch):
    monkeypatch.setenv("AEON_V1_LLM", "0")
    cfg = Config()
    assert cfg.llm_enabled is False


def test_llm_config_defaults():
    cfg = Config()
    assert cfg.llm_provider == "lmstudio"
    assert cfg.llm_model == "local-model"
    assert cfg.llm_temperature == 0.2
    assert cfg.llm_max_tokens == 1200
    assert cfg.llm_timeout_seconds == 60



# ---------------------------------------------------------------------------
# generate_text adapter
# ---------------------------------------------------------------------------

def test_generate_text_returns_none_when_disabled():
    cfg = Config()
    cfg.llm_enabled = False
    assert generate_text("hello", cfg) is None


def test_generate_text_returns_none_when_lmstudio_unavailable():
    """LM Studio connection failures return None — system never crashes."""
    cfg = Config()
    cfg.llm_enabled = True

    with patch("aeon_v1.llm._call_lmstudio_messages", return_value=None):
        result = generate_text("hello", cfg)
    assert result is None


def test_generate_image_description_uses_vision_model_and_no_reasoning(tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(base64_png_bytes())
    cfg = Config(tmp_path)
    cfg.llm_enabled = True
    cfg.llm_provider = "lmstudio"
    cfg.llm_vision_model = "vendor/vision-model"
    cfg.llm_reasoning_effort = "low"

    with patch("aeon_v1.llm._call_lmstudio_messages", return_value="vision ok") as call:
        result = generate_image_description(image, "describe", cfg)

    assert result == "vision ok"
    kwargs = call.call_args.kwargs
    assert kwargs["model"] == "vendor/vision-model"
    assert kwargs["include_reasoning"] is False


def test_generate_image_description_falls_back_to_native_endpoint(tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(base64_png_bytes())
    cfg = Config(tmp_path)
    cfg.llm_enabled = True
    cfg.llm_provider = "lmstudio"
    cfg.llm_vision_model = "vendor/vision-model"

    with patch("aeon_v1.llm._call_lmstudio_messages", return_value=None), \
         patch("aeon_v1.llm._call_lmstudio_native_image", return_value="native ok") as native:
        result = generate_image_description(image, "describe", cfg)

    assert result == "native ok"
    assert native.call_args.kwargs["model"] == "vendor/vision-model"


def test_detect_lmstudio_vision_model_from_loaded_models(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"data":[{"id":"vendor/chat-model"},{"id":"vendor/vision-model-vl"}]}'

    monkeypatch.setattr("aeon_v1.llm.urllib.request.urlopen", lambda *_, **__: FakeResponse())

    assert _detect_lmstudio_vision_model(Config()) == "vendor/vision-model-vl"


def test_resolve_lmstudio_vision_model_prefers_explicit_config(monkeypatch):
    monkeypatch.setattr("aeon_v1.llm._detect_lmstudio_vision_model", lambda _: "vendor/vision-model-vl")
    cfg = Config()
    cfg.llm_vision_model = "explicit-vision"

    assert resolve_lmstudio_vision_model(cfg) == "explicit-vision"


# ---------------------------------------------------------------------------
# No execution primitives in llm.py
# ---------------------------------------------------------------------------

def test_no_execution_primitives_in_llm():
    tree = ast.parse((SRC_DIR / "llm.py").read_text(encoding="utf-8"))
    forbidden = {"subprocess", "os.system", "os.popen", "shutil", "exec", "eval"}
    imports_found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    imports_found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in forbidden:
                imports_found.add(node.module)
    assert imports_found == set(), f"llm.py imports forbidden modules: {imports_found}"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_reflection_prompt_is_non_empty():
    analysis = {
        "sources": [],
        "source_types": {"episodic": 1, "semantic": 0},
        "detected_patterns": ["Tag X repeated"],
        "uncertainty_notes": [],
        "suggested_tasks": ["Review X"],
        "confidence": 0.5,
    }
    prompt = build_reflection_prompt(analysis)
    assert len(prompt) > 100
    assert "vault/core/" in prompt  # safety instruction present
    assert "### What Was Learned" in prompt
    assert "### Suggested Tasks" in prompt


def test_simulation_prompt_is_non_empty():
    task = {"title": "test-task", "description": "Investigate memory leak", "priority": 0.8, "confidence": 0.7}
    prompt = build_simulation_prompt(task)
    assert len(prompt) > 100
    assert "SIMULATION ONLY" in prompt
    assert "### Proposed Action" in prompt
    assert "### Risk Assessment" in prompt


def test_reflection_prompt_includes_safety_rules():
    analysis = {"sources": [], "source_types": {}, "detected_patterns": [],
                "uncertainty_notes": [], "suggested_tasks": [], "confidence": 0.3}
    prompt = build_reflection_prompt(analysis)
    assert "do not invent" in prompt.lower() or "only the information provided" in prompt.lower()
    assert "core memory" in prompt.lower()


# ---------------------------------------------------------------------------
# parse_reflection_sections / parse_simulation_sections
# ---------------------------------------------------------------------------

def test_parse_reflection_sections_full():
    text = """### What Was Learned
- Learned A
- Learned B

### New Patterns Noticed
- Pattern X appears

### Conflicts or Uncertainty
- No conflicts

### Suggested Tasks
- Do thing Y
"""
    sections = parse_reflection_sections(text)
    assert "What Was Learned" in sections
    assert "Learned A" in sections["What Was Learned"]
    assert "New Patterns Noticed" in sections
    assert "Conflicts or Uncertainty" in sections
    assert "Suggested Tasks" in sections


def test_parse_reflection_sections_partial():
    """Missing sections are omitted — no KeyError."""
    text = "### What Was Learned\n- Only this section\n"
    sections = parse_reflection_sections(text)
    assert "What Was Learned" in sections
    assert "New Patterns Noticed" not in sections


def test_parse_simulation_sections_full():
    text = """### Proposed Action
Review the configuration file and update settings.

### Expected Outcome
Configuration will be updated and system will restart cleanly.

### Risk Assessment
- Human review required before any real action is taken.
- Configuration changes may affect other services.
"""
    sections = parse_simulation_sections(text)
    assert "Proposed Action" in sections
    assert "Expected Outcome" in sections
    assert "Risk Assessment" in sections


def test_parse_simulation_sections_empty_response():
    sections = parse_simulation_sections("")
    assert sections == {}


# ---------------------------------------------------------------------------
# Reflection — fallback path (LLM disabled)
# ---------------------------------------------------------------------------

def test_reflection_works_without_llm(tmp_path):
    config = _tmp_config(tmp_path)
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1

    ingest(
        'I learned a critical key insight: "Fallback Memory" is an important concept.',
        config=config,
    )
    result = reflect(config=config)
    ref = result["reflection"]
    assert ref is not None
    content = ref["content"]
    _assert_all_7_sections(content)
    assert ref.get("llm_used") is False
    assert ref.get("llm_model") is None
    assert ref.get("llm_provider") is None


def test_reflection_all_7_sections_always_present(tmp_path):
    config = _tmp_config(tmp_path)
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1
    ingest(
        'I learned a critical key insight: "Section Test" is an important concept.',
        config=config,
    )
    result = reflect(config=config)
    _assert_all_7_sections(result["reflection"]["content"])


# ---------------------------------------------------------------------------
# Reflection — mocked LLM path
# ---------------------------------------------------------------------------

_MOCK_REFLECTION_LLM = """### What Was Learned
- LLM-synthesized insight about memory systems

### New Patterns Noticed
- LLM detected a recurring theme about learning

### Conflicts or Uncertainty
- LLM found no significant conflicts

### Suggested Tasks
- LLM suggests investigating the memory consolidation process
"""


def test_reflection_uses_mocked_llm_output(tmp_path):
    config = _tmp_config(tmp_path)
    config.llm_enabled = True
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1

    ingest(
        'I learned a critical key insight: "LLM Memory" is an important concept.',
        config=config,
    )

    with patch("aeon_v1.reflect.generate_text", return_value=_MOCK_REFLECTION_LLM):
        result = reflect(config=config)

    ref = result["reflection"]
    assert ref is not None
    content = ref["content"]
    _assert_all_7_sections(content)
    assert "LLM-synthesized insight" in content
    assert "LLM detected a recurring theme" in content
    assert ref.get("llm_used") is True
    assert ref.get("llm_model") == config.llm_model
    assert ref.get("llm_provider") == config.llm_provider


def test_reflection_llm_sections_2_6_7_always_rule_based(tmp_path):
    """Section 2 (memories list), 6 (core warning), 7 (quality) are never LLM-generated."""
    config = _tmp_config(tmp_path)
    config.llm_enabled = True
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1

    ingest(
        'I learned a critical key insight: "Rule Based" is an important concept.',
        config=config,
    )

    with patch("aeon_v1.reflect.generate_text", return_value=_MOCK_REFLECTION_LLM):
        result = reflect(config=config)

    content = result["reflection"]["content"]
    assert "Human review required" in content       # Section 6 core warning intact
    assert "**Confidence:**" in content             # Section 7 quality intact


def test_reflection_fallback_when_llm_returns_empty(tmp_path):
    """If LLM returns empty string, rule-based sections are used."""
    config = _tmp_config(tmp_path)
    config.llm_enabled = True
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1

    ingest(
        'I learned a critical key insight: "Empty LLM" is an important concept.',
        config=config,
    )

    with patch("aeon_v1.reflect.generate_text", return_value=""):
        result = reflect(config=config)

    ref = result["reflection"]
    assert ref.get("llm_used") is False
    _assert_all_7_sections(ref["content"])


# ---------------------------------------------------------------------------
# Simulation — fallback path (LLM disabled)
# ---------------------------------------------------------------------------

def test_simulation_works_without_llm(tmp_path):
    config = _tmp_config(tmp_path)
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1

    ingest(
        'I learned a critical key insight: "Sim Test" is an important concept. '
        'Need to review the simulation pipeline.',
        config=config,
    )
    reflect(config=config)

    tasks = TaskStore(config).list_tasks()
    assert tasks, "No tasks created — cannot test simulation"
    result = simulate_action(tasks[0], config=config)

    sim = result["simulation"]
    assert sim["proposed_action"]
    assert sim["expected_outcome"]
    assert isinstance(sim["risks"], list) and len(sim["risks"]) > 0
    assert sim.get("llm_used") is False
    assert sim.get("llm_model") is None
    assert sim.get("llm_provider") is None


# ---------------------------------------------------------------------------
# Simulation — mocked LLM path
# ---------------------------------------------------------------------------

_MOCK_SIM_LLM = """### Proposed Action
Review the memory consolidation pipeline and identify bottlenecks.

### Expected Outcome
A documented analysis of the pipeline performance with actionable improvements.

### Risk Assessment
- Human review required before any real action is taken.
- Changes to the pipeline may affect other memory layers.
"""


def test_simulation_uses_mocked_llm_output(tmp_path):
    config = _tmp_config(tmp_path)
    config.llm_enabled = True
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1

    ingest(
        'I learned a critical key insight: "LLM Sim" is an important concept. '
        'Need to build the next phase.',
        config=config,
    )
    reflect(config=config)

    tasks = TaskStore(config).list_tasks()
    assert tasks

    with patch("aeon_v1.simulate.generate_text", return_value=_MOCK_SIM_LLM):
        result = simulate_action(tasks[0], config=config)

    sim = result["simulation"]
    assert "Review the memory consolidation" in sim["proposed_action"]
    assert "documented analysis" in sim["expected_outcome"]
    assert sim.get("llm_used") is True
    assert sim.get("llm_model") == config.llm_model
    assert sim.get("llm_provider") == config.llm_provider


def test_simulation_always_requires_human_approval(tmp_path):
    """require_human_approval must be True in simulation regardless of LLM."""
    config = _tmp_config(tmp_path)
    config.llm_enabled = True
    config.allow_low_value_reflections = True
    config.min_reflection_sources = 1
    ingest(
        'I learned a critical key insight: "Approval Test" is an important concept. '
        'Need to fix the approval flow.',
        config=config,
    )
    reflect(config=config)
    tasks = TaskStore(config).list_tasks()
    assert tasks
    with patch("aeon_v1.simulate.generate_text", return_value=_MOCK_SIM_LLM):
        result = simulate_action(tasks[0], config=config)
    assert result["simulation"]["required_human_approval"] is True


def test_simulation_no_real_execution_even_with_llm(tmp_path):
    """simulate.py must not import execution primitives even after LLM integration."""
    tree = ast.parse((SRC_DIR / "simulate.py").read_text(encoding="utf-8"))
    forbidden = {"subprocess", "os.system", "os.popen", "shutil", "exec", "eval"}
    imports_found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    imports_found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in forbidden:
                imports_found.add(node.module)
    assert imports_found == set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.memory_path = tmp_path / "memory"
    cfg.vault_path = tmp_path / "vault"
    return cfg


def base64_png_bytes() -> bytes:
    import base64
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )


def _assert_all_7_sections(content: str) -> None:
    sections = [
        "### What Was Learned",
        "### Important Memories Reviewed",
        "### New Patterns Noticed",
        "### Conflicts or Uncertainty",
        "### Suggested Tasks",
        "### Suggested Core Memory Updates",
        "### Reflection Quality",
    ]
    missing = [s for s in sections if s not in content]
    assert missing == [], f"Missing sections in reflection: {missing}"
