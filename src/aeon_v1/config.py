from pathlib import Path
import os
import sys


def _load_env(path: Path) -> None:
    """Load a .env file into os.environ without overwriting existing vars.

    Skipped during pytest runs so tests control their own environment via monkeypatch.
    """
    argv_text = " ".join(sys.argv).lower()
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in argv_text:
        return
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env from project root (two levels up from this file: src/aeon_v1 -> project root)
_load_env(Path(__file__).parent.parent.parent / ".env")


class Config:
    def __init__(self, base_path: Path = Path(".")):
        self.base_path = Path(base_path)
        self.vault_path = self.base_path / "vault"
        self.memory_path = self.base_path / "memory"
        self.importance_threshold = 0.5
        # Placeholder: reflect after this many ingestions (not enforced automatically yet)
        self.reflection_interval = 10
        self.model_provider = "local"
        # Reflection safety: cap how many source memories one reflection pass reviews
        self.max_memories_per_reflection: int = 50
        # Reflection safety: reflections never reflect on prior reflections by default
        self.allow_reflection_on_reflections: bool = False
        # Core memory protection: ingestion/reflection never write to vault/core/.
        # Setting this True requires an explicit human decision.
        self.allow_core_modification: bool = False
        # Layer 2 — reflection quality controls
        self.min_reflection_sources: int = 1
        self.skip_duplicate_reflections: bool = True
        self.allow_low_value_reflections: bool = False
        # Layer 3 — decision and action simulation
        self.enable_real_actions: bool = False
        self.max_pending_tasks: int = 100
        self.duplicate_task_similarity_threshold: float = 0.8
        self.require_human_approval_for_simulation: bool = True
        # Timestamps — UTC is stored in JSON; this timezone is used for Markdown/CLI display.
        self.display_timezone: str = "America/New_York"
        # Layer 5 — tool registry
        self.allow_tool_override: bool = False
        # Layer 6 — orchestrator / agent pool
        self.max_thinking_agents: int = 10
        self.consolidation_similarity_threshold: float = 0.72
        self.max_consolidations_per_pass: int = 5
        # Background consolidation runs from memory creation events, not a clock.
        # Pytest disables it by default so tests opt in explicitly.
        argv_text = " ".join(sys.argv).lower()
        self.enable_background_consolidation: bool = not (
            os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in argv_text
        )
        self.consolidation_trigger_interval: int = 5
        self.consolidation_trigger_memory_types: list[str] = ["raw", "episodic", "semantic"]
        # Layer 4 — optional LLM reasoning
        # Toggle via AEON_V1_LLM=1 environment variable or set directly.
        self.llm_enabled: bool = os.environ.get("AEON_V1_LLM", "0").strip() == "1"
        self.llm_provider: str = os.environ.get("AEON_V1_LLM_PROVIDER", "anthropic")
        self.llm_model: str = os.environ.get("AEON_V1_LLM_MODEL", "claude-sonnet-4-6")
        self.llm_chat_model: str = os.environ.get("AEON_V1_LLM_CHAT_MODEL", self.llm_model)
        self.llm_deep_model: str = os.environ.get("AEON_V1_LLM_DEEP_MODEL", self.llm_model)
        self.llm_search_model: str = os.environ.get("AEON_V1_LLM_SEARCH_MODEL", self.llm_deep_model)
        self.llm_vision_model: str = os.environ.get("AEON_V1_LLM_VISION_MODEL", "")
        self.llm_temperature: float = 0.2
        self.llm_max_tokens: int = int(os.environ.get("AEON_V1_LLM_MAX_TOKENS", "1200"))
        self.llm_timeout_seconds: int = int(os.environ.get("AEON_V1_LLM_TIMEOUT", "60"))
        self.llm_chat_timeout_seconds: int = int(os.environ.get("AEON_V1_LLM_CHAT_TIMEOUT", "30"))
        self.llm_search_timeout_seconds: int = int(os.environ.get("AEON_V1_LLM_SEARCH_TIMEOUT", "12"))
        self.llm_media_timeout_seconds: int = int(os.environ.get("AEON_V1_LLM_MEDIA_TIMEOUT", "120"))
        self.llm_max_attempts: int = int(os.environ.get("AEON_V1_LLM_MAX_ATTEMPTS", "1"))
        self.llm_reasoning_effort: str = os.environ.get("AEON_V1_LLM_REASONING_EFFORT", "low")
        # LM Studio / OpenAI-compatible local server
        self.llm_base_url: str = os.environ.get("AEON_V1_LLM_BASE_URL", "http://localhost:1234/v1")
        # When True, reflect/simulate use tool calling so the LLM queries the
        # memory index agent instead of receiving all memories inlined in the prompt.
        self.llm_tool_calling: bool = os.environ.get("AEON_V1_LLM_TOOL_CALLING", "0").strip() == "1"

    def ensure_dirs(self):
        for subdir in ["core", "raw", "episodic", "semantic", "reflections", "agents", "tasks", "consolidations", "media"]:
            (self.vault_path / subdir).mkdir(parents=True, exist_ok=True)
        for subdir in ["raw", "episodic", "semantic", "reflections", "consolidations", "media"]:
            (self.memory_path / subdir).mkdir(parents=True, exist_ok=True)
        (self.memory_path / "media" / "uploads").mkdir(parents=True, exist_ok=True)
        (self.memory_path / "schemas").mkdir(parents=True, exist_ok=True)
        # Layer 7 — governance directories
        for subdir in ["staging", "approved", "logs", "tool_additions"]:
            (self.memory_path / subdir).mkdir(parents=True, exist_ok=True)

