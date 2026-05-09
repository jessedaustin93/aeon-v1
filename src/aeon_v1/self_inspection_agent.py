"""Read-only self-inspection for Aeon.

This agent answers questions about what Aeon is, what it can do, and how the
local system works by reading committed project files and runtime counts. It is
deliberately separate from memory search: memories describe what happened,
while self-inspection describes the current program.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .runtime import memory_counts


_SELF_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhat\s+are\s+you\b",
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+can\s+you\s+do\b",
        r"\bwhat\s+are\s+your\s+(?:abilities|capabilities|tools)\b",
        r"\bhow\s+do\s+you\s+work\b",
        r"\bhow\s+are\s+you\s+(?:built|made|wired|set\s+up)\b",
        r"\bexplain\s+(?:yourself|your\s+system|your\s+architecture|how\s+you\s+work)\b",
        r"\bread\s+yourself\b",
        r"\binspect\s+yourself\b",
    ]
]

_SOURCE_FILES = [
    "README.md",
    "src/aeon_v1/README.md",
    "scripts/README.md",
    "memory/README.md",
    "vault/README.md",
    "docs/architecture.md",
    "docs/memory_model.md",
    "docs/recursive_learning_loop.md",
    "docs/tools_manifest.md",
    "docs/obsidian.md",
    "docs/setup_from_github.md",
    "pyproject.toml",
]


class SelfInspectionAgent:
    """Answers self-knowledge questions from Aeon's own repo files."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    def is_self_query(self, text: str) -> bool:
        stripped = text.strip()
        return any(pattern.search(stripped) for pattern in _SELF_PATTERNS)

    def handle_chat_query_with_ids(self, text: str) -> Optional[Dict]:
        if not self.is_self_query(text):
            return None
        sources = self._collect_sources()
        reply = self._format_reply(sources)
        return {
            "reply": reply,
            "memory_ids": [f"self:{source['path']}" for source in sources],
        }

    def handle_chat_query(self, text: str) -> Optional[str]:
        result = self.handle_chat_query_with_ids(text)
        return None if result is None else str(result["reply"])

    def _collect_sources(self) -> List[Dict]:
        sources: List[Dict] = []
        for rel in _SOURCE_FILES:
            path = self.config.base_path / rel
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            sources.append({"path": rel, "text": text})
        return sources

    def _format_reply(self, sources: List[Dict]) -> str:
        caps = self._capabilities(sources)
        docs = ", ".join(source["path"] for source in sources[:6])
        counts = memory_counts(self.config)
        count_text = ", ".join(f"{name}={value}" for name, value in counts.items())
        scripts = self._script_names()

        lines = [
            "I checked my own repo files, not just chat memory.",
            "What I am: a local-first recursive AI memory and learning system built on JSON memory records, Markdown vault notes, optional LM Studio models, and a local dashboard/runner.",
            "How I work: I preserve raw inputs, derive episodic and semantic memories, link vault notes, run reflections/consolidation, and use governed agents for tasks, simulation, and write approval.",
            "Current abilities I can see in my files:",
        ]
        lines.extend(f"- {cap}" for cap in caps[:10])
        if scripts:
            lines.append(f"Launch/utility scripts I can see: {', '.join(scripts)}.")
        if count_text:
            lines.append(f"Current memory counts from runtime: {count_text}.")
        lines.append(f"Sources read: {docs}.")
        return "\n".join(lines)

    def _capabilities(self, sources: List[Dict]) -> List[str]:
        readme = self._source_text(sources, "README.md")
        capabilities = self._readme_capabilities(readme)
        if capabilities:
            return capabilities
        return [
            "chat interface",
            "raw, episodic, semantic, reflection, consolidation, and media memory",
            "local dashboard and always-on runner",
            "LM Studio model routing",
            "Obsidian vault integration",
            "governed write approval pipeline",
        ]

    def _readme_capabilities(self, readme: str) -> List[str]:
        capabilities: List[str] = []
        in_table = False
        for line in readme.splitlines():
            if line.strip() == "## Current Capabilities":
                in_table = True
                continue
            if in_table and line.startswith("## "):
                break
            if not in_table or not line.startswith("|"):
                continue
            parts = [part.strip(" `") for part in line.strip().strip("|").split("|")]
            if len(parts) < 3 or parts[0] in {"Area", "---"}:
                continue
            area, status, modules = parts[:3]
            if "Implemented" in status:
                modules = modules.replace("`", "")
                capabilities.append(f"{area} ({modules})")
        return capabilities

    def _source_text(self, sources: List[Dict], rel_path: str) -> str:
        for source in sources:
            if source["path"] == rel_path:
                return str(source["text"])
        return ""

    def _script_names(self) -> List[str]:
        scripts_dir = self.config.base_path / "scripts"
        if not scripts_dir.exists():
            return []
        return sorted(path.name for path in scripts_dir.glob("*.py"))[:8]
