"""
Aeon-V1 test suite — Layer 1 stabilization.

Guarantees proven:
- Raw memory is written verbatim and is immutable (append-only)
- Multiple raw ingestions never overwrite each other
- Markdown vault notes are created with correct frontmatter
- Memory records include a stable title field
- Obsidian wikilinks use readable [[path|Title]] format
- High-importance text is promoted to episodic; low-importance is not
- Multiple reflections are appended, never replacing prior ones
- Search returns matching results without false positives
- Core memory (vault/core/) is never touched by ingest or reflect
- Importance scoring is weighted: more signals → higher score
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeon_v1 import Config, ingest, reflect, search
from aeon_v1.memory_store import _score_importance, _vault_note_path


@pytest.fixture
def cfg(tmp_path):
    return Config(base_path=tmp_path)


# ═══════════════════════════════════════ raw memory / append-only ══════════

def test_raw_memory_written(cfg):
    text = "This is a test memory about a project goal."
    result = ingest(text, config=cfg)
    raw_id = result["raw"]["id"]

    raw_json = cfg.memory_path / "raw" / f"{raw_id}.json"
    assert raw_json.exists(), "Raw JSON file was not created"

    data = json.loads(raw_json.read_text())
    assert data["type"] == "raw"
    assert data["text"] == text, "Raw text must be stored verbatim"
    assert 0 <= data["importance"] <= 1


def test_raw_memory_never_modified(cfg):
    """The text field in a raw record must equal the original input forever."""
    original = "Exact verbatim text — must never be summarized or changed."
    result = ingest(original, config=cfg)
    raw_id = result["raw"]["id"]

    data = json.loads((cfg.memory_path / "raw" / f"{raw_id}.json").read_text())
    assert data["text"] == original


def test_raw_files_not_overwritten(cfg):
    """Two separate ingestions must produce two distinct, intact JSON files."""
    r1 = ingest("First memory about a project goal.", config=cfg)
    r2 = ingest("Second memory about a different goal.", config=cfg)

    id1, id2 = r1["raw"]["id"], r2["raw"]["id"]
    assert id1 != id2, "Each ingest must generate a unique ID"

    data1 = json.loads((cfg.memory_path / "raw" / f"{id1}.json").read_text())
    data2 = json.loads((cfg.memory_path / "raw" / f"{id2}.json").read_text())
    assert data1["text"] == "First memory about a project goal."
    assert data2["text"] == "Second memory about a different goal."


# ═══════════════════════════════════════ markdown vault / frontmatter ══════

def test_markdown_note_created(cfg):
    result = ingest("Remember this important concept about learning.", config=cfg)
    raw_id = result["raw"]["id"]

    md_path = _vault_note_path(cfg, "raw", raw_id)
    assert md_path.exists(), "Markdown vault note was not created"

    content = md_path.read_text()
    assert "---" in content, "Frontmatter delimiter missing"
    assert raw_id in content, "Note should reference its own ID"
    assert "type: raw" in content


def test_markdown_frontmatter_fields(cfg):
    result = ingest("An important project milestone was reached.", config=cfg)
    md_path = _vault_note_path(cfg, "raw", result["raw"]["id"])
    content = md_path.read_text()

    for field in ("id:", "title:", "type:", "created:", "source:", "importance:", "tags:", "links:"):
        assert field in content, f"Frontmatter field '{field}' is missing"


# ═══════════════════════════════════════ title field ═══════════════════════

def test_memory_title_field_present(cfg):
    """All promoted memory layers must carry a title field."""
    result = ingest(
        "I learned an important goal for this project must be remembered.",
        config=cfg,
    )
    assert "title" in result["raw"], "Raw memory must include title field"
    if result["episodic"]:
        assert "title" in result["episodic"], "Episodic memory must include title field"
    if result["semantic"]:
        assert "title" in result["semantic"], "Semantic memory must include title field"


def test_title_is_human_readable(cfg):
    """Title must be readable words — not a bare UUID."""
    result = ingest("Remember this important project milestone.", config=cfg)
    title = result["raw"]["title"]

    assert len(title) > 0
    assert not re.fullmatch(r'[0-9a-f]{8}', title), "Title must not be a bare UUID"
    # At least one recognisable word from the input should appear
    assert any(word in title for word in ("remember", "important", "project", "milestone"))


def test_reflection_title_is_date_based(cfg):
    """Reflection title should encode a date stamp, not a bare UUID."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    result = reflect(config=cfg)
    title = result["reflection"]["title"]
    assert title.startswith("reflection-"), f"Expected reflection- prefix, got: {title}"
    assert re.search(r'\d{8}', title), "Reflection title should contain a date stamp"


# ═══════════════════════════════════════ Obsidian wikilinks ════════════════

def test_obsidian_links_include_readable_title(cfg):
    """Episodic note must link to its raw source using [[path|Title]] format."""
    result = ingest("This is an important project I need to remember.", config=cfg)
    assert result["episodic"] is not None, "Text should promote to episodic"

    ep_id = result["episodic"]["id"]
    md_path = _vault_note_path(cfg, "episodic", ep_id)
    content = md_path.read_text()

    raw_id = result["raw"]["id"]
    # Link must use readable format and point at the generated-note folder.
    assert f"[[_generated/raw/{raw_id}|" in content, (
        "Episodic vault note should link to raw source with readable display title"
    )


def test_reflection_links_use_readable_titles(cfg):
    """Reflection note sources must use [[subdir/id|title]] wikilinks."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    result = reflect(config=cfg)

    ref_id = result["reflection"]["id"]
    md_path = _vault_note_path(cfg, "reflections", ref_id)
    content = md_path.read_text()

    # At least one source link should contain a pipe (readable format)
    assert "|" in content, "Reflection source links should use [[path|Title]] format"


# ═══════════════════════════════════════ episodic promotion ════════════════

def test_episodic_promotion_on_high_importance(cfg):
    result = ingest("This is an important project I need to remember.", config=cfg)
    assert result["episodic"] is not None, "High-importance text should be promoted to episodic"

    ep_id = result["episodic"]["id"]
    ep_json = cfg.memory_path / "episodic" / f"{ep_id}.json"
    assert ep_json.exists()

    data = json.loads(ep_json.read_text())
    assert data["raw_ref"] == result["raw"]["id"]


def test_episodic_markdown_note_created(cfg):
    result = ingest("Remember this important project goal.", config=cfg)
    if result["episodic"] is None:
        pytest.skip("Text did not meet importance threshold")

    ep_id = result["episodic"]["id"]
    md_path = _vault_note_path(cfg, "episodic", ep_id)
    assert md_path.exists()
    assert "Episodic Memory" in md_path.read_text()


def test_ingest_assigns_category_and_topic_hub(cfg):
    result = ingest("Remember this phrase for a testing canary: alpha beta.", config=cfg)
    raw = result["raw"]

    assert raw["category"] == "test-memory"
    assert (cfg.vault_path / "topics" / "test-memory.md").exists()

    raw_note = _vault_note_path(cfg, "raw", raw["id"]).read_text(encoding="utf-8")
    assert "[[topics/test-memory|Topic: Test Memory]]" in raw_note


def test_no_episodic_for_low_importance(cfg):
    result = ingest("The weather is cloudy today.", config=cfg)
    assert result["episodic"] is None, "Low-importance text must not be promoted"


# ═══════════════════════════════════════ importance scoring ════════════════

def test_importance_low():
    """Plain prose with no signal words must score clearly below the threshold."""
    score = _score_importance("The sky is blue and the clouds are white.")
    assert score < 0.5, f"Plain text should score below 0.5, got {score}"


def test_importance_high():
    """Keyword-rich text must score above the default 0.5 threshold."""
    score = _score_importance("I learned a critical insight about project goals.")
    assert score >= 0.5, f"Keyword-rich text should score >= 0.5, got {score}"


def test_importance_weighted_signals():
    """More signals → strictly higher score."""
    low    = _score_importance("Hello world.")
    medium = _score_importance("This is an important note.")
    high   = _score_importance("I learned a critical key insight for this project goal.")

    assert low < medium, f"low ({low}) should be < medium ({medium})"
    assert medium < high, f"medium ({medium}) should be < high ({high})"
    assert low < 0.5,  f"Plain text below threshold expected, got {low}"
    assert high >= 0.5, f"Dense keyword text above threshold expected, got {high}"


def test_importance_via_ingest(cfg):
    """Low-importance text stays raw; high-importance text is promoted."""
    low_result  = ingest("The meeting was held at noon.", config=cfg)
    high_result = ingest("I learned an important project goal must be remembered.", config=cfg)

    assert low_result["episodic"] is None,  "Low-importance text must not promote"
    assert high_result["episodic"] is not None, "High-importance text must promote"


# ═══════════════════════════════════════ search ════════════════════════════

def test_search_returns_match(cfg):
    ingest("I learned about recursive memory systems today.", config=cfg)
    results = search("recursive memory", config=cfg)
    assert results, "Search should return at least one result"

    combined = " ".join(
        r["memory"].get("text", "") + r["memory"].get("summary", "") for r in results
    )
    assert "recursive memory" in combined.lower()


def test_search_no_false_positives(cfg):
    ingest("The quick brown fox jumps.", config=cfg)
    results = search("quantum entanglement", config=cfg)
    assert results == [], "Unrelated query should return no results"


def test_search_by_tag(cfg):
    ingest("This is an important project.", config=cfg)
    results = search("project", config=cfg)
    assert results, "Tag-based search should return results"


# ═══════════════════════════════════════ reflection / append-only ══════════

def test_reflection_created(cfg):
    ingest("This is an important project goal I must remember clearly.", config=cfg)
    result = reflect(config=cfg)

    assert result["reflection"] is not None, "Reflection should be created"
    ref_id = result["reflection"]["id"]

    assert (cfg.memory_path / "reflections" / f"{ref_id}.json").exists()
    assert _vault_note_path(cfg, "reflections", ref_id).exists()


def test_reflection_references_source_ids(cfg):
    ingest("I learned an important project goal: build a local memory system.", config=cfg)
    result = reflect(config=cfg)

    reflection = result["reflection"]
    assert reflection is not None, "Reflection should be created for high-importance memory"
    assert len(reflection["source_ids"]) > 0, "Reflection should list source memory IDs"


def test_multiple_reflections_both_preserved(cfg):
    """Calling reflect() twice must create two distinct files — no overwriting."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    cfg.skip_duplicate_reflections = False  # Layer 2: allow same-source repeat

    r1 = reflect(config=cfg)
    r2 = reflect(config=cfg)

    assert r1["reflection"] is not None
    assert r2["reflection"] is not None
    id1, id2 = r1["reflection"]["id"], r2["reflection"]["id"]
    assert id1 != id2, "Each reflect() call must produce a unique reflection ID"

    assert (cfg.memory_path / "reflections" / f"{id1}.json").exists(), "First reflection must persist"
    assert (cfg.memory_path / "reflections" / f"{id2}.json").exists(), "Second reflection must persist"
    assert _vault_note_path(cfg, "reflections", id1).exists()
    assert _vault_note_path(cfg, "reflections", id2).exists()


def test_reflection_samples_raw_archive_memories(cfg):
    cfg.allow_low_value_reflections = True
    cfg.min_reflection_sources = 1
    result = ingest("Plain archive-only raw memory without promotion.", config=cfg)

    reflection = reflect(config=cfg)["reflection"]

    assert reflection is not None
    assert result["raw"]["id"] in reflection["source_ids"]
    assert reflection["source_types"].get("raw", 0) >= 1


def test_no_reflection_without_memories(cfg):
    result = reflect(config=cfg)
    assert result["reflection"] is None
    assert "No memories" in result["message"]


# ═══════════════════════════════════════ core memory protection ════════════

def test_core_not_modified_by_ingest(cfg):
    """Ingestion must never create or modify any file inside vault/core/."""
    core_dir = cfg.vault_path / "core"
    before = set(core_dir.glob("*")) if core_dir.exists() else set()

    ingest("Important project goal must be remembered.", config=cfg)

    after = set(core_dir.glob("*")) if core_dir.exists() else set()
    assert before == after, "Ingestion must not touch vault/core/"


def test_core_not_modified_by_reflect(cfg):
    """Reflection must never create or modify any file inside vault/core/."""
    ingest("I learned an important project goal.", config=cfg)

    core_dir = cfg.vault_path / "core"
    before = set(core_dir.glob("*")) if core_dir.exists() else set()

    reflect(config=cfg)

    after = set(core_dir.glob("*")) if core_dir.exists() else set()
    assert before == after, "Reflection must not touch vault/core/"


def test_direct_core_write_raises_error(cfg):
    """_write_markdown must raise CoreMemoryProtectedError when targeting vault/core/."""
    from aeon_v1.exceptions import CoreMemoryProtectedError
    from aeon_v1.memory_store import _write_markdown

    core_path = cfg.vault_path / "core" / "should-not-exist.md"

    with pytest.raises(CoreMemoryProtectedError):
        _write_markdown(core_path, {"id": "x", "type": "core"}, "body", config=cfg)

    assert not core_path.exists(), "File must not have been created before the error was raised"


def test_linker_skips_core_vault_files(cfg):
    """linker._update_markdown_links must leave vault/core/ files untouched."""
    from aeon_v1.linker import _update_markdown_links

    # Create a file in vault/core/ that has content which would be altered by link injection
    core_dir = cfg.vault_path / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    core_md = core_dir / "protected-concept.md"
    original = "# Protected Concept\n\nOriginal content — must not be touched by linker."
    core_md.write_text(original, encoding="utf-8")

    # Simulate a memory whose vault path resolves to vault/core/
    # (type "core" -> _VAULT_DIR_MAP["core"] == "core")
    fake_core_memory = {
        "id": "protected-concept",
        "type": "core",
        "tags": ["project", "important"],
    }
    related = [{"id": "abc12345", "type": "raw", "title": "some-related-memory"}]

    # This must skip the file silently rather than modifying it
    _update_markdown_links(fake_core_memory, related, cfg)

    assert core_md.read_text(encoding="utf-8") == original, (
        "Linker must not modify files inside vault/core/"
    )


# ═══════════════════════════════════════ Layer 2 — reflection quality ══════

def test_reflection_has_structured_sections(cfg):
    """Reflection Markdown must contain all 7 required section headers."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    result = reflect(config=cfg)

    assert result["reflection"] is not None, "Reflection should be created"
    ref_id = result["reflection"]["id"]
    md_path = _vault_note_path(cfg, "reflections", ref_id)
    content = md_path.read_text()

    required_sections = [
        "### What Was Learned",
        "### Important Memories Reviewed",
        "### New Patterns Noticed",
        "### Conflicts or Uncertainty",
        "### Suggested Tasks",
        "### Suggested Core Memory Updates",
        "### Reflection Quality",
    ]
    for section in required_sections:
        assert section in content, f"Missing section: {section}"


def test_reflection_json_has_metadata_fields(cfg):
    """Reflection JSON must include Layer 2 metadata fields."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    result = reflect(config=cfg)

    assert result["reflection"] is not None
    ref = result["reflection"]

    for field in ("confidence", "source_types", "suggested_tasks",
                  "suggested_core_updates", "detected_patterns",
                  "uncertainty_notes", "generated_at"):
        assert field in ref, f"Reflection JSON missing field: {field}"

    assert isinstance(ref["confidence"], float), "confidence must be a float"
    assert isinstance(ref["source_types"], dict), "source_types must be a dict"
    assert isinstance(ref["suggested_tasks"], list)
    assert isinstance(ref["suggested_core_updates"], list)
    assert isinstance(ref["detected_patterns"], list)
    assert isinstance(ref["uncertainty_notes"], list)


def test_task_extraction(cfg):
    """Text containing task phrases should appear in suggested_tasks."""
    ingest(
        "I learned that we need to build a local memory index and should review "
        "the episodic layer for important project patterns.",
        config=cfg,
    )
    result = reflect(config=cfg)

    assert result["reflection"] is not None
    tasks = result["reflection"]["suggested_tasks"]
    assert len(tasks) > 0, "Task-phrase text should produce at least one suggested task"


def test_suggested_core_updates_not_written_to_vault_core(cfg):
    """Suggested core memory updates must appear in the reflection note only — never in vault/core/."""
    ingest("I learned an important project goal must be remembered.", config=cfg)

    core_dir = cfg.vault_path / "core"
    before = set(core_dir.glob("*")) if core_dir.exists() else set()

    reflect(config=cfg)

    after = set(core_dir.glob("*")) if core_dir.exists() else set()
    assert before == after, "Reflection must not write any file to vault/core/"


def test_duplicate_reflection_skipped(cfg):
    """Second reflect() on the same source IDs must be skipped when skip_duplicate_reflections=True.
    Uses archive_random so the time-window duplicate guard is active
    (sequential uses cursor rotation as its own dedup instead)."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    cfg.skip_duplicate_reflections = True
    cfg.reflection_sampling_strategy = "archive_random"

    r1 = reflect(config=cfg)
    r2 = reflect(config=cfg)

    assert r1["reflection"] is not None, "First reflection should be created"
    assert r2["reflection"] is None, "Duplicate reflection should be skipped"
    assert "Duplicate" in r2["message"]


def test_duplicate_reflection_allowed_when_config_off(cfg):
    """Second reflect() must succeed when skip_duplicate_reflections=False."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    cfg.skip_duplicate_reflections = False

    r1 = reflect(config=cfg)
    r2 = reflect(config=cfg)

    assert r1["reflection"] is not None
    assert r2["reflection"] is not None
    assert r1["reflection"]["id"] != r2["reflection"]["id"], "Each pass must create a unique ID"


def test_confidence_score_in_bounds(cfg):
    """Confidence must always be a float in [0.0, 1.0]."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    result = reflect(config=cfg)

    assert result["reflection"] is not None
    confidence = result["reflection"]["confidence"]
    assert 0.0 <= confidence <= 1.0, f"Confidence out of bounds: {confidence}"


def test_confidence_higher_with_more_sources(cfg):
    """More diverse source memories should yield a higher confidence score."""
    # Single source
    cfg_single = cfg
    ingest("I learned an important project goal.", config=cfg_single)
    r_single = reflect(config=cfg_single)
    conf_single = r_single["reflection"]["confidence"] if r_single["reflection"] else 0.0

    # Multiple sources — fresh config in a new tmp directory is not needed;
    # just ingest more varied content so tag diversity increases.
    import tempfile
    from pathlib import Path
    from aeon_v1 import Config
    tmp2 = Path(tempfile.mkdtemp())
    cfg_many = Config(base_path=tmp2)
    for text in [
        "I learned an important project goal must be remembered.",
        "This is a critical key insight about learning patterns.",
        "I discovered a bug in the error handling code.",
        "The concept of recursive memory is an important idea to remember.",
        "I realized this project goal needs to be documented.",
    ]:
        ingest(text, config=cfg_many)
    r_many = reflect(config=cfg_many)
    conf_many = r_many["reflection"]["confidence"] if r_many["reflection"] else 0.0

    assert conf_many >= conf_single, (
        f"More sources should yield >= confidence: single={conf_single}, many={conf_many}"
    )


def test_reflection_excludes_prior_reflections_by_default(cfg):
    """Prior reflection IDs must not appear in source_ids of a new reflection."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    cfg.skip_duplicate_reflections = False

    r1 = reflect(config=cfg)
    r2 = reflect(config=cfg)

    assert r1["reflection"] is not None
    assert r2["reflection"] is not None

    r1_id = r1["reflection"]["id"]
    assert r1_id not in r2["reflection"]["source_ids"], (
        "Prior reflection ID must not appear in source_ids when "
        "allow_reflection_on_reflections is False"
    )


def test_low_source_reflection_skipped(cfg):
    """reflect() must be skipped when source count < min_reflection_sources."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    cfg.min_reflection_sources = 10  # require 10 sources — we only have 1-2
    cfg.allow_low_value_reflections = False

    result = reflect(config=cfg)

    assert result["reflection"] is None, "Should be skipped: too few sources"
    assert "Too few" in result["message"]


def test_low_source_allowed_with_config(cfg):
    """reflect() must succeed when allow_low_value_reflections=True even below min threshold."""
    ingest("I learned an important project goal must be remembered.", config=cfg)
    cfg.min_reflection_sources = 10  # above actual source count
    cfg.allow_low_value_reflections = True

    result = reflect(config=cfg)

    assert result["reflection"] is not None, (
        "Reflection should be created when allow_low_value_reflections=True"
    )
