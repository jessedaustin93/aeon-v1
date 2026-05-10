import json

from aeon_v1.config import Config
from aeon_v1.reflection_maintenance import minimize_reflections


def _write_reflection(cfg: Config, idx: int, confidence: float = 0.5, raw_count: int = 20) -> None:
    rid = f"r{idx:07d}"
    record = {
        "id": rid,
        "title": f"reflection-{idx}",
        "type": "reflection",
        "created": f"2026-05-10T00:{idx:02d}:00+00:00",
        "content": f"## Recursive Reflection\n\nLearning task memory pattern {idx % 3}",
        "source_ids": [f"s{idx}"],
        "source_types": {"raw": raw_count, "episodic": 20 - raw_count},
        "confidence": confidence,
        "suggested_tasks": [],
        "suggested_core_updates": [],
        "detected_patterns": [],
        "tags": ["memory"],
    }
    (cfg.memory_path / "reflections" / f"{rid}.json").write_text(json.dumps(record), encoding="utf-8")
    vault_dir = cfg.vault_path / "_generated" / "reflections"
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / f"{rid}.md").write_text(f"# reflection {idx}", encoding="utf-8")


def test_minimize_reflections_archives_excess_and_keeps_summary(tmp_path):
    cfg = Config(tmp_path)
    cfg.ensure_dirs()
    for idx in range(12):
        _write_reflection(cfg, idx, confidence=0.5 + idx / 100, raw_count=20 if idx < 8 else 5)

    result = minimize_reflections(cfg, keep_active=5, keep_recent=2)

    assert result["archived_count"] == 7
    assert result["kept_count"] == 5
    assert len(list((cfg.memory_path / "reflections").glob("*.json"))) == 5
    assert len(list((cfg.memory_path / "reflections" / "archive").glob("*/*.json"))) == 7
    assert len(list((cfg.vault_path / "_generated" / "reflections" / "archive").glob("*/*.md"))) == 7
    summaries = list((cfg.memory_path / "consolidations").glob("*.json"))
    assert summaries
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["source"] == "reflection_maintenance"
    assert summary["archived_count"] == 7
