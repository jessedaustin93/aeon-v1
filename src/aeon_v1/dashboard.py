"""Local Aeon control dashboard."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chat_cli import ChatTurn, build_response, retrieve_context, _load_recent_turns
from .config import Config
from .ingest import ingest
from .launcher_config import load_launcher_config
from .linker import link_memories
from .media import ingest_image_data_url
from .memory_index_agent import MemoryIndexAgent
from .search_agent import SearchAgent
from .self_inspection_agent import SelfInspectionAgent
from .runtime import (
    launcher_status_path,
    memory_counts,
    process_alive,
    read_json,
    runner_status_path,
    runner_stop_path,
    write_json,
)
from .time_utils import utc_now_iso


class DashboardController:
    def __init__(self, base_path: Path, launcher_config_path: Optional[Path] = None) -> None:
        self.base_path = base_path.resolve()
        self.config = Config(self.base_path)
        self.config.llm_chat_timeout_seconds = min(self.config.llm_chat_timeout_seconds, 6)
        self.config.importance_threshold = 0.2
        self.config.ensure_dirs()
        self.launcher_config = load_launcher_config(self.base_path, launcher_config_path)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._transcript_path = self.config.memory_path / "chat" / "dashboard_transcript.jsonl"
        self._chat_history: List[ChatTurn] = _load_recent_turns(self._transcript_path, limit=8)
        self._index_agent = MemoryIndexAgent(self.config)
        self._search_agent = SearchAgent(self.config)
        self._self_agent = SelfInspectionAgent(self.config)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ status

    def status(self) -> Dict[str, Any]:
        runner = read_json(runner_status_path(self.config))
        runner_pid = int(runner.get("pid", 0) or 0)
        runner_alive = bool(runner_pid and process_alive(runner_pid))
        runner["alive"] = runner_alive and runner.get("state") == "running"

        return {
            "updated_at": utc_now_iso(),
            "base_path": str(self.base_path),
            "memory_counts": memory_counts(self.config),
            "runner": runner,
            "lm_studio": self._lm_studio_status(),
            "obsidian": self._obsidian_status(),
            "dashboard": {
                "state": "running",
                "pid": os.getpid(),
            },
        }

    # ---------------------------------------------------------------- controls

    def start_runner(self) -> Dict[str, Any]:
        with self._lock:
            existing = self._processes.get("runner")
            if existing and existing.poll() is None:
                return {"ok": True, "message": "runner already started by dashboard"}

            runner_cfg = self.launcher_config.get("runner", {})
            command = [
                sys.executable,
                str(self.base_path / "scripts" / "aeon_runner.py"),
                "--base-path",
                str(self.base_path),
                "--poll-seconds",
                str(runner_cfg.get("poll_seconds", 5)),
                "--link-every-passes",
                str(runner_cfg.get("link_every_passes", 12)),
            ]
            popen = subprocess.Popen(
                command,
                cwd=str(self.base_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_creation_flags(),
            )
            self._processes["runner"] = popen
            self._write_launcher_status("runner", "started", pid=popen.pid)
            return {"ok": True, "message": "runner started", "pid": popen.pid}

    def stop_runner(self) -> Dict[str, Any]:
        runner_stop_path(self.config).write_text(utc_now_iso(), encoding="utf-8")
        proc = self._processes.get("runner")
        if proc and proc.poll() is None:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.terminate()
        self._write_launcher_status("runner", "stop_requested")
        return {"ok": True, "message": "runner stop requested"}

    def start_lm_studio(self) -> Dict[str, Any]:
        app_cfg = self.launcher_config.get("lm_studio", {})
        if not app_cfg.get("enabled", True):
            return {"ok": False, "message": "LM Studio launch is disabled"}
        return self._start_external("lm_studio", app_cfg, _default_lm_studio_command())

    def start_obsidian(self) -> Dict[str, Any]:
        app_cfg = self.launcher_config.get("obsidian", {})
        if not app_cfg.get("enabled", True):
            return {"ok": False, "message": "Obsidian launch is disabled"}

        command = str(app_cfg.get("command", "")).strip()
        if command:
            return self._start_external("obsidian", app_cfg, "")

        vault_name = str(app_cfg.get("vault_name", "")).strip()
        vault_path = self.base_path / str(app_cfg.get("vault_path", "vault"))
        if vault_name:
            webbrowser.open(f"obsidian://open?vault={urllib.parse.quote(vault_name)}")
            return {"ok": True, "message": "opened Obsidian vault URL"}
        if vault_path.exists():
            webbrowser.open(f"obsidian://open?path={urllib.parse.quote(str(vault_path))}")
            return {"ok": True, "message": "opened Obsidian path URL"}
        return {"ok": False, "message": "vault path does not exist"}

    def quit_all(self) -> Dict[str, Any]:
        result = {"runner": self.stop_runner()}
        for name in ("lm_studio", "obsidian"):
            proc = self._processes.get(name)
            if proc and proc.poll() is None:
                proc.terminate()
                result[name] = {"ok": True, "message": "terminate requested"}
        self._write_launcher_status("dashboard", "quit_requested")
        return {"ok": True, "message": "quit requested", "components": result}

    def chat(
        self,
        text: str,
        image_filename: str = "",
        image_data_url: str = "",
    ) -> Dict[str, Any]:
        text = text.strip()
        has_image = bool(image_data_url.strip())
        if not text and not has_image:
            return {"ok": False, "message": "empty chat message"}

        media_record = None
        if has_image:
            media_result = ingest_image_data_url(
                data_url=image_data_url,
                filename=image_filename or "upload.png",
                source="aeon-dashboard-chat",
                config=self.config,
                prompt=(
                    f"Describe this image in relation to the user's message: {text}"
                    if text else
                    "Describe this image for Aeon's local memory and chat response."
                ),
            )
            media_record = media_result.get("media")

        image_context = ""
        if media_record:
            image_context = (
                f"\n\nAttached image: {media_record.get('original_name', image_filename)}\n"
                f"Image analysis status: {media_record.get('analysis_status')}\n"
                f"Image description: {media_record.get('description')}"
            )
        elif has_image:
            image_context = "\n\nAttached image: upload received, but image analysis failed."

        effective_text = (text or "Please look at this image.") + image_context

        self_result = None if media_record else self._self_agent.handle_chat_query_with_ids(effective_text)
        search_result = None if media_record or self_result else self._search_agent.handle_chat_query_with_ids(effective_text)
        user_memory_id = None
        if self_result is None and search_result is None:
            user_memory_id = self._ingest_chat_text(f"User: {effective_text}")
        memories = retrieve_context(effective_text, self.config, limit=5)
        llm_used = False
        if media_record and media_record.get("analysis_status") == "complete":
            response = self._image_chat_response(text, media_record)
        elif self_result is not None:
            response = str(self_result["reply"])
        elif search_result is not None:
            response = str(search_result["reply"])
        else:
            response = build_response(
                user_text=effective_text,
                memories=memories,
                history=self._chat_history[-4:],
                config=self.config,
                index_agent=self._index_agent,
            )
            llm_used = not response.startswith("[local]")
        assistant_memory_id = None
        if self_result is None and search_result is None:
            assistant_memory_id = self._ingest_chat_text(f"Aeon: {response}")
        self._link_safely()

        turn = ChatTurn(
            user=text,
            assistant=response,
            memory_ids=list(self_result.get("memory_ids", [])) if self_result else list(search_result.get("memory_ids", [])) if search_result else [
                mid for mid in (user_memory_id, assistant_memory_id) if mid
            ],
            llm_used=llm_used,
        )
        self._chat_history.append(turn)
        self._append_dashboard_transcript(turn)
        return {
            "ok": True,
            "reply": response,
            "memory_ids": turn.memory_ids,
            "llm_used": turn.llm_used,
            "media": media_record,
        }

    def upload_image(self, filename: str, data_url: str, prompt: str = "") -> Dict[str, Any]:
        result = ingest_image_data_url(
            data_url=data_url,
            filename=filename or "upload.png",
            source="aeon-dashboard",
            config=self.config,
            prompt=prompt.strip() or None,
        )
        self._link_safely()
        return {"ok": result.get("media") is not None, **result}

    # ---------------------------------------------------------------- internals

    def _ingest_chat_text(self, text: str) -> Optional[str]:
        try:
            result = ingest(text, source="aeon-dashboard", config=self.config)
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

    def _image_chat_response(self, text: str, media_record: Dict[str, Any]) -> str:
        description = str(media_record.get("description", "")).strip()
        if text:
            return f"I looked at the image. {description}"
        return description or "I looked at the image and stored it in media memory."

    def _append_dashboard_transcript(self, turn: ChatTurn) -> None:
        try:
            self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with self._transcript_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "at": utc_now_iso(),
                    "user": turn.user,
                    "assistant": turn.assistant,
                    "memory_ids": turn.memory_ids,
                    "llm_used": turn.llm_used,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _start_external(self, name: str, app_cfg: Dict[str, Any], fallback_command: str) -> Dict[str, Any]:
        with self._lock:
            existing = self._processes.get(name)
            if existing and existing.poll() is None:
                return {"ok": True, "message": f"{name} already started by dashboard"}
            command = str(app_cfg.get("command", "") or fallback_command).strip()
            if not command:
                return {"ok": False, "message": f"no command configured for {name}"}
            args = _command_args(command)
            try:
                popen = subprocess.Popen(
                    args,
                    cwd=str(self.base_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_creation_flags(),
                )
            except OSError as exc:
                return {"ok": False, "message": str(exc)}
            self._processes[name] = popen
            self._write_launcher_status(name, "started", pid=popen.pid)
            return {"ok": True, "message": f"{name} started", "pid": popen.pid}

    def _lm_studio_status(self) -> Dict[str, Any]:
        app_cfg = self.launcher_config.get("lm_studio", {})
        base_url = str(app_cfg.get("base_url", "http://localhost:1234/v1")).rstrip("/")
        status: Dict[str, Any] = {"configured": bool(app_cfg.get("enabled", True)), "base_url": base_url}
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=1.5) as response:
                status["server"] = "online"
                payload = json.loads(response.read().decode("utf-8"))
                status["models"] = [m.get("id") for m in payload.get("data", []) if m.get("id")]
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            status["server"] = "offline"
            status["models"] = []
        proc = self._processes.get("lm_studio")
        status["launched_by_dashboard"] = bool(proc and proc.poll() is None)
        return status

    def _obsidian_status(self) -> Dict[str, Any]:
        app_cfg = self.launcher_config.get("obsidian", {})
        vault_path = self.base_path / str(app_cfg.get("vault_path", "vault"))
        proc = self._processes.get("obsidian")
        return {
            "configured": bool(app_cfg.get("enabled", True)),
            "vault_path": str(vault_path),
            "vault_exists": vault_path.exists(),
            "obsidian_config_exists": (vault_path / ".obsidian").exists(),
            "launched_by_dashboard": bool(proc and proc.poll() is None),
        }

    def _write_launcher_status(self, component: str, state: str, **extra: Any) -> None:
        data = read_json(launcher_status_path(self.config))
        data[component] = {"state": state, "updated_at": utc_now_iso(), **extra}
        write_json(launcher_status_path(self.config), data)


def run_dashboard(
    base_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    launcher_config_path: Optional[Path] = None,
    open_browser: bool = False,
) -> ThreadingHTTPServer:
    controller = DashboardController(base_path, launcher_config_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(_dashboard_html())
                return
            if self.path == "/api/status":
                self._send_json(controller.status())
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/api/chat":
                payload = self._read_json_body()
                self._send_json(controller.chat(
                    text=str(payload.get("text", "")),
                    image_filename=str(payload.get("image_filename", "")),
                    image_data_url=str(payload.get("image_data_url", "")),
                ))
                return
            if self.path == "/api/media/image":
                payload = self._read_json_body()
                self._send_json(controller.upload_image(
                    filename=str(payload.get("filename", "upload.png")),
                    data_url=str(payload.get("data_url", "")),
                    prompt=str(payload.get("prompt", "")),
                ))
                return

            routes = {
                "/api/start/runner": controller.start_runner,
                "/api/stop/runner": controller.stop_runner,
                "/api/start/lm-studio": controller.start_lm_studio,
                "/api/start/obsidian": controller.start_obsidian,
                "/api/quit": controller.quit_all,
            }
            handler = routes.get(self.path)
            if handler is None:
                self.send_error(404)
                return
            result = handler()
            self._send_json(result)
            if self.path == "/api/quit":
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *_: object) -> None:
            return

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: Dict[str, Any]) -> None:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length).decode("utf-8")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    server = ThreadingHTTPServer((host, port), Handler)
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    server.serve_forever()
    return server


def _dashboard_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aeon Control</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --panel-2: #eef2f5;
      --text: #18202a;
      --muted: #657282;
      --line: #d9e0e7;
      --accent: #256d7b;
      --accent-2: #8b5f34;
      --ok: #24764b;
      --warn: #a45d18;
      --bad: #a13a3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .shell { max-width: 1180px; margin: 0 auto; padding: 28px; }
    header { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 22px; }
    h1 { font-size: 30px; line-height: 1.1; margin: 0 0 8px; letter-spacing: 0; }
    .sub { color: var(--muted); margin: 0; font-size: 14px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 7px;
      padding: 9px 12px;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button.danger { color: #fff; background: var(--bad); border-color: var(--bad); }
    button:hover { filter: brightness(0.97); }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .panel h2 { margin: 0 0 12px; font-size: 15px; line-height: 1.2; }
    .status { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); }
    .dot { width: 8px; height: 8px; border-radius: 99px; background: var(--muted); }
    .dot.ok { background: var(--ok); }
    .dot.warn { background: var(--warn); }
    .dot.bad { background: var(--bad); }
    .counts { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
    .count { background: var(--panel-2); border-radius: 7px; padding: 10px; min-width: 0; }
    .count strong { display: block; font-size: 24px; line-height: 1; margin-bottom: 5px; }
    .count span { color: var(--muted); font-size: 12px; }
    dl { display: grid; grid-template-columns: 140px 1fr; gap: 8px 12px; margin: 0; font-size: 13px; }
    dt { color: var(--muted); }
    dd { margin: 0; overflow-wrap: anywhere; }
    code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
    .log { white-space: pre-wrap; min-height: 42px; color: var(--muted); font-size: 13px; }
    .chat { display: grid; gap: 12px; }
    .messages {
      min-height: 260px;
      max-height: 420px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg {
      max-width: 78%;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .msg.user { align-self: flex-end; background: var(--accent); color: #fff; }
    .msg.aeon { align-self: flex-start; background: var(--panel-2); color: var(--text); }
    .composer { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: end; }
    .composer-fields { display: grid; gap: 8px; }
    .composer textarea {
      min-height: 46px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 11px;
      font: inherit;
      font-size: 14px;
      color: var(--text);
      background: #fff;
    }
    .composer input[type="file"] {
      position: absolute;
      inline-size: 1px;
      block-size: 1px;
      opacity: 0;
      pointer-events: none;
    }
    .attach-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .attach-button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      width: fit-content;
      border: 1px dashed var(--accent);
      background: #eef7f8;
      color: var(--accent);
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    .attach-name { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .media-form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: end;
    }
    .media-form input[type="file"], .media-form input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      font: inherit;
      font-size: 13px;
      background: #fff;
    }
    .media-meta {
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
      white-space: pre-wrap;
    }
    @media (max-width: 820px) {
      .shell { padding: 18px; }
      header { display: block; }
      .actions { justify-content: flex-start; margin-top: 14px; }
      .span-4, .span-6, .span-8 { grid-column: span 12; }
      .counts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      dl { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
      .media-form { grid-template-columns: 1fr; }
      .msg { max-width: 92%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Aeon Control</h1>
        <p class="sub">Local launcher, status board, and clean shutdown surface.</p>
      </div>
      <div class="actions">
        <button class="primary" onclick="post('/api/start/runner')">Start Runner</button>
        <button onclick="post('/api/start/lm-studio')">Start LM Studio</button>
        <button onclick="post('/api/start/obsidian')">Start Obsidian</button>
        <button onclick="post('/api/stop/runner')">Stop Runner</button>
        <button class="danger" onclick="post('/api/quit')">Quit Aeon</button>
      </div>
    </header>
    <section class="grid">
      <article class="panel span-12">
        <h2>Chat</h2>
        <div class="chat">
          <div id="messages" class="messages">
            <div class="msg aeon">Aeon dashboard is online. Type here when you want the local memory chat without opening the terminal.</div>
          </div>
          <form id="chatForm" class="composer">
            <div class="composer-fields">
              <textarea id="chatInput" placeholder="Talk to Aeon..." aria-label="Chat with Aeon"></textarea>
              <div class="attach-row">
                <label class="attach-button" for="chatImageInput">Attach Image</label>
                <span id="chatImageName" class="attach-name">No image attached</span>
                <input id="chatImageInput" type="file" accept="image/png,image/jpeg,image/webp,image/gif" aria-label="Attach image for Aeon">
              </div>
            </div>
            <button class="primary" type="submit">Send</button>
          </form>
        </div>
      </article>
      <article class="panel span-12">
        <h2>Image Input</h2>
        <form id="imageForm" class="media-form">
          <div>
            <input id="imageInput" type="file" accept="image/png,image/jpeg,image/webp,image/gif" aria-label="Upload image for Aeon">
            <input id="imagePrompt" type="text" placeholder="Optional: what should Aeon look for?" aria-label="Image analysis prompt">
          </div>
          <button class="primary" type="submit">Analyze Image</button>
        </form>
        <div id="mediaResult" class="media-meta">Load a vision model in LM Studio, then upload an image here.</div>
      </article>
      <article class="panel span-4">
        <h2>Runner</h2>
        <div id="runnerStatus" class="status"><span class="dot"></span><span>checking</span></div>
        <dl id="runnerDetails"></dl>
      </article>
      <article class="panel span-4">
        <h2>LM Studio</h2>
        <div id="lmStatus" class="status"><span class="dot"></span><span>checking</span></div>
        <dl id="lmDetails"></dl>
      </article>
      <article class="panel span-4">
        <h2>Obsidian</h2>
        <div id="obsidianStatus" class="status"><span class="dot"></span><span>checking</span></div>
        <dl id="obsidianDetails"></dl>
      </article>
      <article class="panel span-8">
        <h2>Memory Counts</h2>
        <div id="counts" class="counts"></div>
      </article>
      <article class="panel span-4">
        <h2>Dashboard</h2>
        <dl id="dashboardDetails"></dl>
      </article>
      <article class="panel span-12">
        <h2>Activity</h2>
        <div id="log" class="log">Ready.</div>
      </article>
    </section>
  </main>
  <script>
    const log = document.getElementById('log');
    const messages = document.getElementById('messages');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatImageInput = document.getElementById('chatImageInput');
    const chatImageName = document.getElementById('chatImageName');
    const imageForm = document.getElementById('imageForm');
    const imageInput = document.getElementById('imageInput');
    const imagePrompt = document.getElementById('imagePrompt');
    const mediaResult = document.getElementById('mediaResult');
    async function post(path) {
      const res = await fetch(path, { method: 'POST' });
      const data = await res.json();
      log.textContent = JSON.stringify(data, null, 2);
      refresh();
    }
    async function refresh() {
      const res = await fetch('/api/status');
      const data = await res.json();
      render(data);
    }
    function addMessage(role, text) {
      const div = document.createElement('div');
      div.className = `msg ${role}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }
    chatForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = chatInput.value.trim();
      const imageFile = chatImageInput.files && chatImageInput.files[0];
      if (!text && !imageFile) return;
      chatInput.value = '';
      chatImageInput.value = '';
      chatImageName.textContent = 'No image attached';
      addMessage('user', imageFile ? `${text || 'Image attached'}\n[image: ${imageFile.name}]` : text);
      addMessage('aeon', 'Thinking...');
      const thinking = messages.lastElementChild;
      try {
        const imageDataUrl = imageFile ? await readFileAsDataUrl(imageFile) : '';
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            image_filename: imageFile ? imageFile.name : '',
            image_data_url: imageDataUrl
          })
        });
        const data = await res.json();
        thinking.textContent = data.reply || data.message || 'No reply returned.';
        log.textContent = JSON.stringify({ chat: data }, null, 2);
        refresh();
      } catch (err) {
        thinking.textContent = `Chat failed: ${err}`;
      }
    });
    chatImageInput.addEventListener('change', () => {
      const file = chatImageInput.files && chatImageInput.files[0];
      chatImageName.textContent = file ? file.name : 'No image attached';
    });
    function readFileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }
    imageForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const file = imageInput.files && imageInput.files[0];
      if (!file) {
        mediaResult.textContent = 'Choose an image first.';
        return;
      }
      mediaResult.textContent = 'Analyzing image...';
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const res = await fetch('/api/media/image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              filename: file.name,
              data_url: reader.result,
              prompt: imagePrompt.value
            })
          });
          const data = await res.json();
          if (data.media) {
            mediaResult.textContent = `${data.media.analysis_status}: ${data.media.description}`;
          } else {
            mediaResult.textContent = data.error || 'Image upload failed.';
          }
          log.textContent = JSON.stringify({ image: data }, null, 2);
          refresh();
        } catch (err) {
          mediaResult.textContent = `Image upload failed: ${err}`;
        }
      };
      reader.readAsDataURL(file);
    });
    function statusEl(id, ok, label, warn=false) {
      const el = document.getElementById(id);
      const cls = ok ? 'ok' : (warn ? 'warn' : 'bad');
      el.innerHTML = `<span class="dot ${cls}"></span><span>${label}</span>`;
    }
    function dl(id, rows) {
      document.getElementById(id).innerHTML = rows.map(([k,v]) => `<dt>${k}</dt><dd>${v ?? ''}</dd>`).join('');
    }
    function render(data) {
      const runner = data.runner || {};
      statusEl('runnerStatus', runner.alive, runner.alive ? 'running' : (runner.state || 'stopped'), runner.state === 'stopped');
      dl('runnerDetails', [
        ['pid', runner.pid || ''],
        ['heartbeat', runner.heartbeat_at || runner.updated_at || ''],
        ['passes', runner.passes || 0],
        ['last link', runner.last_link_at || ''],
        ['last error', runner.last_error || ''],
      ]);
      const lm = data.lm_studio || {};
      statusEl('lmStatus', lm.server === 'online', lm.server || 'unknown');
      dl('lmDetails', [
        ['base url', `<code>${lm.base_url || ''}</code>`],
        ['models', (lm.models || []).join(', ') || 'none detected'],
        ['dashboard launch', lm.launched_by_dashboard ? 'yes' : 'no'],
      ]);
      const obs = data.obsidian || {};
      statusEl('obsidianStatus', obs.vault_exists, obs.vault_exists ? 'vault found' : 'vault missing', obs.vault_exists);
      dl('obsidianDetails', [
        ['vault', `<code>${obs.vault_path || ''}</code>`],
        ['config', obs.obsidian_config_exists ? 'found' : 'not found'],
        ['dashboard launch', obs.launched_by_dashboard ? 'yes' : 'no'],
      ]);
      const counts = data.memory_counts || {};
      document.getElementById('counts').innerHTML = Object.entries(counts).map(([k,v]) => (
        `<div class="count"><strong>${v}</strong><span>${k}</span></div>`
      )).join('');
      dl('dashboardDetails', [
        ['pid', data.dashboard?.pid || ''],
        ['base path', `<code>${data.base_path || ''}</code>`],
        ['updated', data.updated_at || ''],
      ]);
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>"""


def _command_args(command: str) -> list[str]:
    if os.name == "nt" and command.lower().endswith(".exe") and Path(command).exists():
        return [command]
    return shlex.split(command, posix=os.name != "nt")


def _default_lm_studio_command() -> str:
    for candidate in ("lm-studio", "lmstudio", "LM Studio.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    if os.name == "nt":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if not root:
                continue
            path = Path(root) / "LM Studio" / "LM Studio.exe"
            if path.exists():
                return str(path)
    return ""


def _creation_flags() -> int:
    if os.name == "nt":
        return subprocess.CREATE_NO_WINDOW
    return 0
