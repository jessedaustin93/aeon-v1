"""Action simulation engine for Aeon-V1 Layer 3.

simulate_action() produces a structured simulation record describing what
*would* happen if the task were acted upon — and stores it as JSON + Markdown.

SAFETY GUARANTEE
================
This module NEVER imports or calls:
  subprocess, os.system, os.popen, shutil, exec, eval,
  PowerShell, bash, shell, or any network/execution primitive.

All output is written to local files only. enable_real_actions is checked
on every call and raises RuntimeError if True (the value is always False
in the shipped Config — changing it does nothing because no execution path
exists to enable).
"""
import json
import re
from typing import Dict, List, Optional

from .config import Config
from .llm import (
    build_simulation_prompt,
    build_simulation_prompt_sparse,
    generate_text,
    generate_with_memory,
    parse_simulation_sections,
)
from .memory_store import _generate_id, _vault_note_path, _wikilink
from .tasks import TaskStore
from .time_utils import local_date_time_string, utc_now_iso
from .tool_calls import ToolCallStore
from .tools import ToolRegistry

_DESTRUCTIVE_SIGNALS = ("delete", "remove", "drop", "clear", "wipe", "purge")
_EXTERNAL_SIGNALS    = ("deploy", "push", "publish", "release", "ship")
_NETWORK_SIGNALS     = ("api", "request", "call", "fetch", "send", "http", "url")

# Maps registered tool names to the description keywords that signal a match.
# Only tools present in the ToolRegistry are considered; unregistered names are skipped.
_TOOL_SIGNALS: Dict[str, tuple] = {
    "file_read":       ("read", "load", "open", "view", "inspect"),
    "file_write":      ("write", "save", "create", "update", "append", "generate"),
    "command_preview": ("run", "execute", "command", "shell", "script", "invoke"),
}


class SimulationStore:
    """Read/write simulation records from memory/simulations/ and vault/simulations/."""

    def __init__(self, config: Config):
        self.config = config
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        (self.config.memory_path / "simulations").mkdir(parents=True, exist_ok=True)
        _vault_note_path(self.config, "simulations", ".keep").parent.mkdir(parents=True, exist_ok=True)

    def store(self, simulation: Dict) -> Dict:
        sim_id = simulation["id"]
        (self.config.memory_path / "simulations" / f"{sim_id}.json").write_text(
            json.dumps(simulation, indent=2), encoding="utf-8"
        )
        self._write_markdown(simulation)
        return simulation

    def update_feedback(self, sim_id: str, feedback: str) -> Optional[Dict]:
        """Update the feedback field of a simulation record in place.

        Args:
            sim_id:   ID of the simulation to update.
            feedback: One of "success", "failure", "unknown".

        Returns the updated record, or None if the simulation was not found.
        """
        if feedback not in {"success", "failure", "unknown"}:
            raise ValueError(f"feedback must be 'success', 'failure', or 'unknown'; got {feedback!r}")
        path = self.config.memory_path / "simulations" / f"{sim_id}.json"
        if not path.exists():
            return None
        sim = json.loads(path.read_text(encoding="utf-8"))
        sim["feedback"] = feedback
        path.write_text(json.dumps(sim, indent=2), encoding="utf-8")
        self._write_markdown(sim)
        return sim

    def list_simulations(self) -> List[Dict]:
        sim_dir = self.config.memory_path / "simulations"
        if not sim_dir.exists():
            return []
        sims = []
        for f in sim_dir.glob("*.json"):
            try:
                sims.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return sorted(sims, key=lambda s: s.get("created_at", ""))

    def _write_markdown(self, sim: Dict) -> None:
        md_path = _vault_note_path(self.config, "simulations", sim["id"])
        fm_lines = ["---"]
        for key, val in [
            ("id",                      sim["id"]),
            ("type",                    "simulation"),
            ("task_id",                 sim["task_id"]),
            ("task_title",              sim["task_title"]),
            ("created_at",              sim["created_at"]),
            ("required_human_approval", sim["required_human_approval"]),
            ("feedback",                sim.get("feedback", "unknown")),
        ]:
            fm_lines.append(f"{key}: {val}")
        fm_lines.append("source_links:")
        for lnk in sim.get("source_links", []):
            fm_lines.append(f"  - {lnk}")
        fm_lines.append("---")

        risks_md = (
            "\n".join(f"- {r}" for r in sim.get("risks", []))
            or "- None identified"
        )
        tool_call = sim.get("tool_call")
        if tool_call:
            tool_call_md = f"```json\n{json.dumps(tool_call, indent=2)}\n```"
        else:
            tool_call_md = "_No matching tool found in registry._"

        display_tz = self.config.display_timezone
        local_ts = local_date_time_string(sim["created_at"], display_tz)
        task_link = _wikilink("tasks", sim["task_id"], sim["task_title"])
        body = (
            f"# Simulation — {sim['task_title']}\n\n"
            f"**Created:** {local_ts}\n\n"
            f"**Task:** {task_link}\n\n"
            f"**Proposed Action:** {sim['proposed_action']}\n\n"
            f"**Tool Call:**\n\n{tool_call_md}\n\n"
            f"**Expected Outcome:** {sim['expected_outcome']}\n\n"
            f"**Risks:**\n{risks_md}\n\n"
            f"**Required Human Approval:** `{sim['required_human_approval']}`\n\n"
            "> **SIMULATION ONLY** — no system commands were executed.\n"
            "> All actions above are proposed and require human review.\n\n"
            "[[Simulations]] | [[Decisions]] | [[Tasks]]"
        )
        md_path.write_text("\n".join(fm_lines) + "\n\n" + body, encoding="utf-8")


def simulate_action(task: Dict, config: Optional[Config] = None) -> Dict:
    """Produce and store a simulation record for a task.

    The function writes files only. It never calls any execution primitive.

    Args:
        task:   A task dict as returned by TaskStore.
        config: Optional Config — defaults to Config().

    Returns:
        {"simulation": simulation_record}
    """
    from .write_guard import assert_write_authorized
    assert_write_authorized("simulate_action")
    if config is None:
        config = Config()

    # Safety check — belt-and-suspenders even though no execution path exists.
    if config.enable_real_actions:
        raise RuntimeError(
            "enable_real_actions is True, but no real execution path is implemented. "
            "Aeon-V1 is a simulation-only system. Do not set enable_real_actions = True."
        )

    sim_store = SimulationStore(config)
    task_store = TaskStore(config)

    sim_id   = _generate_id()
    now      = utc_now_iso()
    desc     = task.get("description", "")
    title    = task.get("title", task.get("id", "unknown"))
    task_link = _wikilink("tasks", task["id"], title)

    # --- LLM attempt for planning fields (falls back to rule-based on any failure) ---
    llm_meta: Dict = {"llm_used": False, "llm_model": None, "llm_provider": None}
    llm_sections: Dict = {}

    if config.llm_enabled:
        if config.llm_tool_calling:
            from .memory_index_agent import MemoryIndexAgent
            agent = MemoryIndexAgent(config)
            llm_text = generate_with_memory(build_simulation_prompt_sparse(task), agent, config)
        else:
            llm_text = generate_text(build_simulation_prompt(task), config)
        if llm_text:
            llm_sections = parse_simulation_sections(llm_text)
            if llm_sections:
                llm_meta = {
                    "llm_used":     True,
                    "llm_model":    config.llm_deep_model if config.llm_tool_calling else config.llm_model,
                    "llm_provider": config.llm_provider,
                }

    proposed_action  = llm_sections.get("Proposed Action") or _propose_action(desc)
    expected_outcome = llm_sections.get("Expected Outcome") or _expected_outcome(title, desc)
    risks            = _risks_from_llm(llm_sections.get("Risk Assessment"), desc, config)
    tool_call        = _match_tool_call(desc, config)

    # Persist matched tool call as its own traceable record.
    tool_call_id: Optional[str] = None
    if tool_call is not None:
        call_store   = ToolCallStore(config)
        call_record  = call_store.create(
            tool_call=tool_call,
            simulation_id=sim_id,
            task_id=task["id"],
            task_title=title,
        )
        tool_call_id = call_record["id"]

    simulation: Dict = {
        "id":                      sim_id,
        "task_id":                 task["id"],
        "task_title":              title,
        "proposed_action":         proposed_action,
        "expected_outcome":        expected_outcome,
        "risks":                   risks,
        "tool_call":               tool_call,
        "tool_call_id":            tool_call_id,
        "required_human_approval": config.require_human_approval_for_simulation,
        "feedback":                "unknown",
        "created_at":              now,
        "source_links":            [task_link],
        "llm_used":                llm_meta["llm_used"],
        "llm_model":               llm_meta["llm_model"],
        "llm_provider":            llm_meta["llm_provider"],
    }

    sim_store.store(simulation)
    task_store.update_status(task["id"], "simulated")

    return {"simulation": simulation}


# ---------------------------------------------------------------------------
# Tool call mapping — matches task description to a registered tool schema.
# ---------------------------------------------------------------------------

def _match_tool_call(description: str, config: Config) -> Optional[Dict]:
    """Map a task description to the best-matching registered tool.

    Scores each tool in _TOOL_SIGNALS by counting keyword hits in the
    description. Returns a structured call dict for the top scorer, or None
    if no registered tool scores above zero.
    """
    registry  = ToolRegistry(config)
    desc_lower = description.lower()

    best_name:  Optional[str] = None
    best_score: int           = 0

    for tool_name, signals in _TOOL_SIGNALS.items():
        tool = registry.get(tool_name)
        if tool is None or not tool.enabled:
            continue
        score = sum(1 for s in signals if re.search(rf"\b{re.escape(s)}\b", desc_lower))
        if score > best_score:
            best_score = score
            best_name  = tool_name

    if best_name is None:
        return None

    tool = registry.get(best_name)
    return {
        "tool":               best_name,
        "arguments":          _extract_arguments(description, best_name),
        "matched_by":         "keyword",
        "approval_required":  tool.approval_required,
    }


def _extract_arguments(description: str, tool_name: str) -> Dict:
    """Extract inferred arguments from the task description for the matched tool."""
    if tool_name in ("file_read", "file_write"):
        path = _extract_path(description)
        return {"path": path} if path else {}
    if tool_name == "command_preview":
        cmd = _extract_command(description)
        return {"command": cmd} if cmd else {}
    return {}


def _extract_path(text: str) -> Optional[str]:
    """Return the first file path found in text, or None."""
    # Prefer explicitly quoted paths
    m = re.search(r'["\']([^"\']+\.[a-zA-Z]{1,6})["\']', text)
    if m:
        return m.group(1)
    # Bare word with a recognisable file extension
    m = re.search(r'\b([\w./\\-]+\.[a-z]{2,6})\b', text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _extract_command(text: str) -> Optional[str]:
    """Return the most likely shell command found in text, or None."""
    # Backtick-delimited is the most explicit signal
    m = re.search(r'`([^`]+)`', text)
    if m:
        return m.group(1).strip()
    # After an action verb
    m = re.search(
        r'\b(?:run|execute|invoke|call)\b\s+(?:the\s+)?["\']?([\w][\w\s.-]{1,60}?)["\']?'
        r'(?=\s*[,.]|\s*$)',
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Rule-based fallbacks — used when LLM is disabled or unavailable.
# LLM integration is in simulate_action() via llm.generate_text().
# ---------------------------------------------------------------------------

def _propose_action(description: str) -> str:
    desc_lower = description.lower()
    if any(s in desc_lower for s in _DESTRUCTIVE_SIGNALS):
        return f"[DESTRUCTIVE — human approval required] {description[:120]}"
    if any(s in desc_lower for s in _EXTERNAL_SIGNALS):
        return f"[EXTERNAL-FACING — human approval required] {description[:120]}"
    return f"Review and act on: {description[:120]}"


def _expected_outcome(title: str, description: str) -> str:
    return (
        f"Task '{title}' is addressed according to its description. "
        "Relevant memory layers are updated as needed by subsequent ingestion."
    )


def _risks_from_llm(llm_text: Optional[str], description: str, config: Config) -> List[str]:
    """Parse LLM Risk Assessment text into a risk list, or fall back to rule-based."""
    if llm_text:
        risks = [
            line.lstrip("-•* ").strip()
            for line in llm_text.splitlines()
            if line.strip()
        ]
        risks = [r for r in risks if r]
        if risks:
            if config.require_human_approval_for_simulation:
                approval = "Human review required before any real action is taken."
                if not any("human review" in r.lower() for r in risks):
                    risks = [approval] + risks
            return risks
    return _estimate_risks(description, config)


def _estimate_risks(description: str, config: Config) -> List[str]:
    risks = []
    if config.require_human_approval_for_simulation:
        risks.append("Human review required before any real action is taken.")
    desc_lower = description.lower()
    if any(s in desc_lower for s in _DESTRUCTIVE_SIGNALS):
        risks.append("Destructive operation — verify exact scope before proceeding.")
    if any(s in desc_lower for s in _EXTERNAL_SIGNALS):
        risks.append("External-facing action — confirm environment and recipients.")
    if any(s in desc_lower for s in _NETWORK_SIGNALS):
        risks.append("Network operation — ensure authorization and rate limits are respected.")
    if not risks:
        risks.append("No specific risk signals detected. Standard human review recommended.")
    return risks
