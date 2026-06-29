"""Interactive terminal chat interface for Aeon-V1.

This module is the human-facing front door: a simple chat loop that feels like a
normal AI app while quietly using Aeon's local memory, search, optional LLM, and
maintenance hooks in the background.

It does not bypass Layer 7. It writes ordinary conversational memories through
trusted local ingestion, and it does not execute commands or commit governed
agent proposals.
"""
from __future__ import annotations

import argparse
import cmd
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import Config
from .conversation import ConversationTracker
from .ingest import ingest
from .linker import link_memories
from .llm import generate_chat, generate_music_chat, generate_text, generate_with_memory
from .memory_index_agent import MemoryIndexAgent
from .mesh_dispatch import manage_music
from .orchestrator import Orchestrator
from .reflect import reflect
from .search import search
from .search_agent import SearchAgent
from .self_inspection_agent import SelfInspectionAgent
from .shared_vault import load_startup_context, search_shared_vault
from .time_utils import local_now_string


WELCOME = """
Aeon-V1 Terminal
Local memory online. Type naturally, or use /help for commands.
""".strip()

SYSTEM_PROMPT = """You are Aeon, a local AI memory assistant. You speak through a local terminal.

Voice rules - follow these strictly:
- Talk like a person, not a document. Short sentences, natural tone.
- Keep replies to 1-3 sentences unless the operator explicitly asks for more detail.
- No markdown: no headers (##), no bullet lists, no bold (**text**), no dashes as list items.
- Never pad with preamble like "Great question!" or "Here's what I'll do:".
- If you remember something relevant, weave it in naturally - don't quote memory records.
- Be honest when you don't know something. Don't make things up to fill space.
- If asked to recall a specific fact and retrieved memory does not contain it, say you cannot find it instead of guessing.
- Do not claim actions were executed. If something needs approval, say so in one sentence.
"""

MUSIC_SYSTEM_PROMPT = """You are Aeon's internal music-library planning role for the operator's lawful personal library.
The operator sees one Aeon identity; never present yourself as a separate agent or model.
Scope is the local song pipeline only: resolve releases, acquire through Lidarr/slskd,
tag and organize with beets, deduplicate, and publish into Navidrome.
Return a concise plain-text interpretation and proposed next action.
Do not claim anything was downloaded, changed, deleted, or executed.
Any action must be handed to Agent Mesh and remain subject to operator approval.
Planning governed actions is allowed. Lidarr resolves and queues releases; slskd
downloads from Soulseek; beets tags, deduplicates, and organizes; Navidrome indexes
the finished library. Never claim slskd converts or tags files.
Do not manage movies, television, books, or unrelated files.
"""

_MUSIC_ACTION_RE = re.compile(
    r"\b(?:plan|queue|grab|download|acquire|import|add|tag|retag|organize|sort|dedupe|"
    r"deduplicate|scan|refresh|remove|delete|upgrade|replace|find)\b",
    re.IGNORECASE,
)
_MUSIC_OBJECT_RE = re.compile(
    r"\b(?:music|song|songs|track|tracks|album|albums|artist|artists|playlist|"
    r"lidarr|navidrome|soulseek|slskd|beets|flac|mp3)\b",
    re.IGNORECASE,
)


def is_music_management_request(text: str) -> bool:
    """Return True only for explicit song-library management requests."""
    return bool(_MUSIC_ACTION_RE.search(text) and _MUSIC_OBJECT_RE.search(text))


@dataclass
class ChatOptions:
    base_path: Path = Path(".")
    source: str = "aeon-chat"
    no_ingest: bool = False
    auto_link: bool = False
    auto_tick: bool = False
    reflect_every: int = 0
    memory_limit: int = 5
    transcript_path: Optional[Path] = None


@dataclass
class ChatTurn:
    user: str
    assistant: str
    memory_ids: List[str] = field(default_factory=list)
    llm_used: bool = False


class TerminalChatApp(cmd.Cmd):
    """Small cmd-based chat shell for Aeon."""

    intro = WELCOME
    prompt = "aeon> "

    def __init__(self, config: Config, options: ChatOptions):
        super().__init__()
        self.config = config
        self.options = options
        # Chat conversations are episodic by nature, so lower the threshold so every
        # meaningful exchange gets promoted, not just keyword-heavy notes.
        self.config.importance_threshold = 0.2
        self.index_agent = MemoryIndexAgent(config)
        self.search_agent = SearchAgent(config)
        self.self_agent = SelfInspectionAgent(config)
        self.turns: List[ChatTurn] = _load_recent_turns(options.transcript_path, limit=6)
        self.turn_count = 0
        self.tracker = ConversationTracker(config=config)
        self.config.ensure_dirs()

    # ------------------------------------------------------------------ input

    def default(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        print("aeon is thinking...")
        turn = self.handle_chat(text)
        print_wrapped(turn.assistant)

    def emptyline(self) -> None:
        return

    # ---------------------------------------------------------------- commands

    def do_help(self, arg: str) -> None:  # noqa: D401 - cmd module convention
        """Show available commands."""
        print(
            textwrap.dedent(
                """
                Commands:
                  /help              show this help
                  /status            show memory + LLM status
                  /memory <query>     search local memory
                  /reflect           run one reflection pass now
                  /tick              run one orchestrator tick now
                  /transcript        show where this session is being logged
                  /music <proposal>  dispatch an accepted music action to Agent Mesh
                  /exit              leave the chat

                You can also just type normally. Aeon will ingest the turn,
                search memory, answer with the configured LLM when available,
                and fall back to a local memory summary when no LLM is enabled.
                """
            ).strip()
        )

    def do_status(self, arg: str) -> None:
        """Show current interface status."""
        status = {
            "base_path": str(self.config.base_path),
            "llm_enabled": self.config.llm_enabled,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "llm_chat_model": self.config.llm_chat_model,
            "llm_deep_model": self.config.llm_deep_model,
            "llm_tool_calling": self.config.llm_tool_calling,
            "auto_link": self.options.auto_link,
            "auto_tick": self.options.auto_tick,
            "reflect_every": self.options.reflect_every,
            "turns": self.turn_count,
            "master_vault_enabled": self.config.master_vault_enabled,
            "master_vault_path": str(self.config.master_vault_path or ""),
        }
        print(json.dumps(status, indent=2))

    def do_memory(self, arg: str) -> None:
        """Search local memory. Usage: /memory recursive learning"""
        query = arg.strip()
        if not query:
            print("Usage: /memory <query>")
            return
        results = search(query, config=self.config)[: self.options.memory_limit]
        if not results:
            print("No matching memory found.")
            return
        for result in results:
            mem = result.get("memory", {})
            print(f"- [{result.get('match_type')}] {mem.get('id', 'unknown')}: {memory_preview(mem)}")

    def do_reflect(self, arg: str) -> None:
        """Run one reflection pass."""
        result = reflect(self.config)
        reflection = result.get("reflection")
        if reflection:
            print(f"Reflection written: {reflection['id']}")
            created = result.get("tasks_created", [])
            if created:
                print(f"Tasks created: {len(created)}")
        else:
            print(result.get("message", "No reflection written."))

    def do_tick(self, arg: str) -> None:
        """Run one orchestrator tick."""
        summary = Orchestrator(self.config).tick()
        print(json.dumps(summary, indent=2, default=str))

    def do_transcript(self, arg: str) -> None:
        """Show transcript location, if enabled."""
        if self.options.transcript_path:
            print(str(self.options.transcript_path))
        else:
            print("Transcript logging is off for this session.")

    def do_music(self, arg: str) -> None:
        """Accept a planned music action and dispatch it as an audited Agent
        Mesh task on T3610. Usage: /music grab the new Sleep Token album in FLAC

        Typing this command IS your explicit acceptance of the proposal. The hub
        records it as a pending approval; nothing runs until you approve it there.
        """
        proposal = arg.strip()
        if not proposal:
            print("Usage: /music <accepted proposal>")
            return
        result = manage_music(proposal, accepted=True, config=self.config)
        print(result.get("detail") or result.get("reason") or result.get("status", ""))
        if result.get("approval_id"):
            print(f"Agent Mesh approval id: {result['approval_id']}")

    def do_exit(self, arg: str) -> bool:
        """Exit Aeon chat."""
        self._close_tracker_safely()
        print("Aeon chat closed.")
        return True

    def do_quit(self, arg: str) -> bool:
        """Exit Aeon chat."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:  # Ctrl+Z/Ctrl+D
        print()
        return self.do_exit(arg)

    # cmd dispatch treats slash commands as unknown syntax; normalize them.
    def onecmd(self, line: str):
        if line.startswith("/"):
            line = line[1:]
        return super().onecmd(line)

    # ------------------------------------------------------------------ chat

    def handle_chat(self, user_text: str) -> ChatTurn:
        self.turn_count += 1
        self.tracker.add_turn("user", user_text)

        self_result = self.self_agent.handle_chat_query_with_ids(user_text)
        search_result = None if self_result else self.search_agent.handle_chat_query_with_ids(user_text)
        user_memory_id = None
        if not self.options.no_ingest and self_result is None and search_result is None:
            user_memory_id = self._ingest_safely(f"User: {user_text}")

        llm_used = False
        if self_result is not None:
            response = str(self_result["reply"])
        elif search_result is not None:
            response = str(search_result["reply"])
        else:
            memories = retrieve_context(user_text, self.config, self.options.memory_limit)
            response = build_response(
                user_text=user_text,
                memories=memories,
                history=self.turns[-4:],
                config=self.config,
                index_agent=self.index_agent,
            )
            llm_used = not response.startswith(local_fallback_prefix())

        self.tracker.add_turn("aeon", response)

        assistant_memory_id = None
        if not self.options.no_ingest and self_result is None and search_result is None:
            assistant_memory_id = self._ingest_safely(f"Aeon: {response}")

        if self.options.auto_link:
            self._link_safely()

        if self.options.auto_tick:
            self._tick_safely()
        elif self.options.reflect_every and self.turn_count % self.options.reflect_every == 0:
            self._reflect_safely()

        turn = ChatTurn(
            user=user_text,
            assistant=response,
            memory_ids=list(self_result.get("memory_ids", [])) if self_result else list(search_result.get("memory_ids", [])) if search_result else [
                mid for mid in (user_memory_id, assistant_memory_id) if mid
            ],
            llm_used=llm_used,
        )
        self.turns.append(turn)
        self._append_transcript(turn)
        return turn

    def _close_tracker_safely(self) -> None:
        try:
            self.tracker.close(store=True)
        except Exception:
            pass

    def _ingest_safely(self, text: str) -> Optional[str]:
        try:
            result = ingest(text, source=self.options.source, config=self.config)
            return (
                (result.get("semantic") or {}).get("id")
                or (result.get("episodic") or {}).get("id")
                or (result.get("raw") or {}).get("id")
            )
        except Exception:
            return None

    def _link_safely(self) -> None:
        try:
            link_memories(config=self.config)
        except Exception:
            pass

    def _tick_safely(self) -> None:
        try:
            Orchestrator(self.config).tick()
        except Exception:
            pass

    def _reflect_safely(self) -> None:
        try:
            reflect(self.config)
        except Exception:
            pass

    def _append_transcript(self, turn: ChatTurn) -> None:
        path = self.options.transcript_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "at": local_now_string(self.config.display_timezone),
                    "user": turn.user,
                    "assistant": turn.assistant,
                    "memory_ids": turn.memory_ids,
                    "llm_used": turn.llm_used,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass


def build_response(
    user_text: str,
    memories: List[Dict],
    history: Iterable[ChatTurn],
    config: Config,
    index_agent: MemoryIndexAgent,
) -> str:
    core = load_core_context(config)
    shared_core = load_startup_context(config, max_chars_per_note=6_000)
    if shared_core:
        core = f"{core}\n\n{shared_core}" if core else shared_core
    if is_music_management_request(user_text):
        # Streamlined path: when auto-approve is on, the request itself dispatches a
        # governed, audited task to the music station instead of only planning.
        if config.mesh_auto_approve:
            outcome = manage_music(user_text, accepted=True, config=config)
            return strip_markdown(outcome.get("detail", outcome.get("reason", "Working on it.")))
        music_text = generate_music_chat(
            [
                {"role": "system", "content": MUSIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            config=config,
        )
        if music_text:
            return strip_markdown(music_text.strip())
    if config.llm_tool_calling:
        prompt = build_chat_prompt(user_text, memories, history, core=core)
        llm_text = generate_with_memory(prompt, index_agent=index_agent, config=config)
    else:
        llm_text = generate_chat(build_chat_messages(user_text, memories, history, core=core), config=config)
    if llm_text:
        return strip_markdown(llm_text.strip())
    return fallback_response(user_text, memories, llm_enabled=config.llm_enabled)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so responses read as plain conversational text."""
    lines = []
    for line in text.splitlines():
        # Drop header lines entirely.
        if re.match(r"^#{1,6}\s+", line):
            line = re.sub(r"^#{1,6}\s+", "", line)
        # Strip leading list markers (- item, * item, 1. item)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        # Strip bold/italic markers
        line = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)
        line = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", line)
        # Strip inline code
        line = re.sub(r"`(.+?)`", r"\1", line)
        lines.append(line)
    # Collapse runs of blank lines to single blank
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result.strip()


def build_chat_prompt(user_text: str, memories: List[Dict], history: Iterable[ChatTurn], core: str = "") -> str:
    memory_block = format_memories(memories)
    history_block = format_history(history)
    return f"""{SYSTEM_PROMPT}

Core identity and rules:
{core or '- None loaded.'}

Recent conversation:
{history_block or '- No prior turns in this session.'}

Relevant local memory:
{memory_block or '- No matching local memory found.'}

User message:
{user_text}

Reply as Aeon. 1-3 sentences max unless the operator asks for more. No markdown, no headers, no bullet points. Talk like a person."""


def build_chat_messages(user_text: str, memories: List[Dict], history: Iterable[ChatTurn], core: str = "") -> List[Dict]:
    context_parts: List[str] = []
    if core:
        context_parts.append(f"Core identity and rules:\n{core}")
    history_block = format_history(history)
    memory_block = format_memories(memories)
    if history_block:
        context_parts.append(f"Recent conversation:\n{history_block}")
    if memory_block:
        context_parts.append(f"Relevant memory and shared context:\n{memory_block}")

    user_content = user_text
    if context_parts:
        user_content = "\n\n".join(context_parts) + f"\n\nUser message:\n{user_text}"

    return [
        {
            "role": "system",
            "content": (
                "You are Aeon, a local AI memory assistant. Talk like a person - short, warm, direct. "
                "No markdown, no headers, no bullet lists. Keep replies to 1-3 sentences unless asked for more. "
                "Do not claim actions were executed. If asked to recall a specific fact and memory does not contain it, say you cannot find it instead of guessing."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def retrieve_context(query: str, config: Config, limit: int) -> List[Dict]:
    search_limit = max(limit, 10)
    results = SearchAgent(config).results(
        query,
        memory_types=["semantic", "episodic", "consolidations", "reflections", "media", "raw"],
        limit=search_limit,
    )
    local = diversify_results(results, limit)
    # Shared context receives a separate, bounded allowance. Retrieval never
    # imports it into Aeon's local memory or reflection pipeline.
    shared_limit = min(3, max(1, limit // 2)) if limit > 0 else 0
    return local + search_shared_vault(query, config, limit=shared_limit)


def diversify_results(results: List[Dict], limit: int) -> List[Dict]:
    """Keep chat context broad so one noisy layer cannot crowd out everything."""
    if limit <= 0:
        return []

    selected: List[Dict] = []
    seen_ids: set = set()
    preferred_types = ["semantic", "episodic", "consolidations", "reflections", "media", "raw"]

    for memory_type in preferred_types:
        for result in results:
            mem = result.get("memory", {})
            mem_id = mem.get("id", "")
            result_type = mem.get("type") or str(result.get("match_type", "")).split("/")[-1]
            if result_type != memory_type or not mem_id or mem_id in seen_ids:
                continue
            selected.append(result)
            seen_ids.add(mem_id)
            break
        if len(selected) >= limit:
            return selected

    for result in results:
        mem_id = result.get("memory", {}).get("id", "")
        if mem_id and mem_id not in seen_ids:
            selected.append(result)
            seen_ids.add(mem_id)
        if len(selected) >= limit:
            break

    return selected


def load_core_context(config: Config) -> str:
    """Load all vault/core/*.md files into a single context block."""
    core_dir = config.vault_path / "core"
    if not core_dir.exists():
        return ""
    lines: List[str] = []
    for f in sorted(core_dir.glob("*.md")):
        if f.name.startswith(".") or f.name == "PROTECTED.md":
            continue
        try:
            content = f.read_text(encoding="utf-8").strip()
            if content:
                lines.append(content)
        except Exception:
            pass
    return "\n\n".join(lines)


def format_memories(results: List[Dict]) -> str:
    lines: List[str] = []
    for result in results:
        mem = result.get("memory", {})
        lines.append(f"- {mem.get('id', 'unknown')} [{result.get('match_type', 'memory')}]: {memory_preview(mem)}")
    return "\n".join(lines)


def format_history(history: Iterable[ChatTurn]) -> str:
    lines: List[str] = []
    for turn in history:
        lines.append(f"User: {turn.user}")
        lines.append(f"Aeon: {turn.assistant}")
    return "\n".join(lines)


def memory_preview(memory: Dict) -> str:
    for field_name in ("summary", "description", "concept", "content", "text", "title"):
        value = memory.get(field_name)
        if value:
            return compact(str(value), 160)
    return compact(str(memory), 160)


def compact(text: str, limit: int) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    cutoff = max(0, limit - 3)
    shortened = one_line[:cutoff].rstrip()
    if cutoff < len(one_line) and one_line[cutoff] != " " and " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened + "..."


def fallback_response(user_text: str, memories: List[Dict], llm_enabled: bool = False) -> str:
    reason = (
        "LM Studio did not return a usable answer before the local timeout, "
        "so I am giving you the local-memory view instead."
        if llm_enabled else
        "LLM mode is off, so I am giving you the local-memory view instead of a generated answer."
    )
    if memories:
        memory_lines = "\n".join(
            f"- {memory_preview(result.get('memory', {}))}" for result in memories[:3]
        )
        return (
            f"{local_fallback_prefix()} I stored that and found a few nearby memories:\n"
            f"{memory_lines}\n\n"
            f"{reason}"
        )
    return (
        f"{local_fallback_prefix()} I stored that. I do not have a close memory match yet, "
        f"and {reason}"
    )


def local_fallback_prefix() -> str:
    return "[local]"


def print_wrapped(text: str) -> None:
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            print()
        else:
            print(textwrap.fill(paragraph, width=88, replace_whitespace=False))


def _load_recent_turns(transcript_path: Optional[Path], limit: int = 6) -> List[ChatTurn]:
    """Read the last `limit` turns from the transcript file to restore conversation history."""
    if transcript_path is None or not transcript_path.exists():
        return []
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
        recent = [l for l in lines if l.strip()][-limit:]
        turns = []
        for line in recent:
            entry = json.loads(line)
            turns.append(ChatTurn(
                user=entry.get("user", ""),
                assistant=entry.get("assistant", ""),
                memory_ids=entry.get("memory_ids", []),
                llm_used=entry.get("llm_used", False),
            ))
        return turns
    except Exception:
        return []


def parse_args(argv: Optional[List[str]] = None) -> ChatOptions:
    parser = argparse.ArgumentParser(description="Open the Aeon-V1 terminal chat interface.")
    parser.add_argument("--base-path", default=".", help="Repo root containing memory/ and vault/.")
    parser.add_argument("--source", default="aeon-chat", help="Source label for ingested chat turns.")
    parser.add_argument("--no-ingest", action="store_true", help="Do not store chat turns in memory.")
    parser.add_argument("--auto-link", action="store_true", help="Run link_memories after each turn. Usually the background runner handles this.")
    parser.add_argument("--auto-tick", action="store_true", help="Run one Orchestrator.tick() after each turn.")
    parser.add_argument(
        "--reflect-every",
        type=int,
        default=10,
        help="Run reflect() every N chat turns. Default 10.",
    )
    parser.add_argument("--memory-limit", type=int, default=5, help="Relevant memories to include per turn.")
    parser.add_argument(
        "--transcript",
        default="memory/chat/transcript.jsonl",
        help="JSONL transcript path relative to base path. Use 'off' to disable.",
    )
    args = parser.parse_args(argv)

    base_path = Path(args.base_path).resolve()
    transcript = None
    if args.transcript.lower() != "off":
        transcript_path = Path(args.transcript)
        transcript = transcript_path if transcript_path.is_absolute() else base_path / transcript_path

    return ChatOptions(
        base_path=base_path,
        source=args.source,
        no_ingest=args.no_ingest,
        auto_link=args.auto_link,
        auto_tick=args.auto_tick,
        reflect_every=max(0, args.reflect_every),
        memory_limit=max(1, args.memory_limit),
        transcript_path=transcript,
    )


def main(argv: Optional[List[str]] = None) -> None:
    options = parse_args(argv)
    config = Config(base_path=options.base_path)
    TerminalChatApp(config, options).cmdloop()


if __name__ == "__main__":
    main()
