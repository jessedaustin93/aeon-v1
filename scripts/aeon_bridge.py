#!/usr/bin/env python3
"""Single-shot Aeon bridge adapter for Agent Mesh command mode.

Reads a prompt from stdin, strips the Agent Mesh Vault protocol header so Aeon
sees only the actual user message, processes one chat turn through Aeon's memory
stack, and writes the plain-text response to stdout then exits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeon_v1.chat_cli import ChatOptions, TerminalChatApp
from aeon_v1.config import Config

AEON_PATH = Path(__file__).resolve().parent.parent
_VAULT_MARKER = "\n\n[Agent Mesh Master Vault protocol]"


def _strip_vault_protocol(text: str) -> str:
    idx = text.find(_VAULT_MARKER)
    return text[:idx].strip() if idx != -1 else text.strip()


def main() -> None:
    prompt = _strip_vault_protocol(sys.stdin.read())
    if not prompt:
        print("Aeon: no input.")
        return

    config = Config(base_path=AEON_PATH)
    config.ensure_dirs()
    options = ChatOptions(
        base_path=AEON_PATH,
        auto_link=False,
        auto_tick=False,
        reflect_every=0,
        memory_limit=5,
        transcript_path=AEON_PATH / "memory" / "chat" / "mesh-transcript.jsonl",
    )
    app = TerminalChatApp(config, options)
    turn = app.handle_chat(prompt)
    print(turn.assistant)


if __name__ == "__main__":
    main()
