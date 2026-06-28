"""LM Studio adapter concurrency and model routing tests."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

from aeon_v1 import Config
from aeon_v1.llm import generate_chat, generate_music_chat, generate_text, generate_with_memory
from aeon_v1.memory_index_agent import MemoryIndexAgent


class _FakeLMStudioHandler(BaseHTTPRequestHandler):
    delay_seconds = 0.0
    reject_tools = False
    active = 0
    max_active = 0
    payloads: List[Dict] = []
    lock = threading.Lock()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        with self.lock:
            type(self).payloads.append(payload)
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)

        try:
            if type(self).reject_tools and payload.get("tools"):
                encoded = b'{"error":"tools unsupported"}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            time.sleep(type(self).delay_seconds)
            body = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": f"ok:{payload.get('model', '')}",
                        },
                    }
                ]
            }
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        finally:
            with self.lock:
                type(self).active -= 1

    def log_message(self, format: str, *args) -> None:
        return


def test_lmstudio_requests_overlap_when_called_in_parallel():
    server = _start_fake_lmstudio(delay_seconds=0.25)
    config = _lmstudio_config(server)

    started = threading.Barrier(5)

    def call(i: int) -> Optional[str]:
        started.wait(timeout=5)
        return generate_chat([{"role": "user", "content": f"hello {i}"}], config)

    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(call, range(5)))
    finally:
        server.shutdown()
        server.server_close()
    elapsed = time.perf_counter() - start

    assert all(result == "ok:chat-model" for result in results)
    assert _FakeLMStudioHandler.max_active > 1
    assert elapsed < 1.0


def test_lmstudio_semaphore_caps_in_flight_requests_at_ten():
    server = _start_fake_lmstudio(delay_seconds=0.35)
    config = _lmstudio_config(server)

    started = threading.Barrier(12)

    def call(i: int) -> Optional[str]:
        started.wait(timeout=5)
        return generate_chat([{"role": "user", "content": f"hello {i}"}], config)

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(call, range(12)))
    finally:
        server.shutdown()
        server.server_close()

    assert _FakeLMStudioHandler.max_active <= 10
    assert results.count(None) == 2
    assert sum(result == "ok:chat-model" for result in results) == 10


def test_lmstudio_model_routing_uses_chat_base_and_deep_models():
    server = _start_fake_lmstudio()
    config = _lmstudio_config(server)
    index_agent = MemoryIndexAgent(config)

    try:
        assert generate_chat([{"role": "user", "content": "chat"}], config) == "ok:chat-model"
        assert generate_text("base", config) == "ok:base-model"
        assert generate_with_memory("deep", index_agent, config) == "ok:deep-model"
    finally:
        server.shutdown()
        server.server_close()

    models = [payload["model"] for payload in _FakeLMStudioHandler.payloads]
    assert models == ["chat-model", "base-model", "deep-model"]


def test_music_role_uses_its_own_model_and_base_url():
    primary = _start_fake_lmstudio()
    music = _start_fake_lmstudio()
    config = _lmstudio_config(primary)
    config.llm_music_model = "music-model"
    config.llm_music_base_url = f"http://127.0.0.1:{music.server_address[1]}/v1"
    config.llm_music_timeout_seconds = 5
    try:
        result = generate_music_chat([{"role": "user", "content": "tag this album"}], config)
    finally:
        primary.shutdown(); primary.server_close()
        music.shutdown(); music.server_close()
    assert result == "ok:music-model"


def test_lmstudio_tool_calling_falls_back_to_deep_model_without_tools():
    server = _start_fake_lmstudio(reject_tools=True)
    config = _lmstudio_config(server)
    index_agent = MemoryIndexAgent(config)

    try:
        assert generate_with_memory("deep", index_agent, config) == "ok:deep-model"
    finally:
        server.shutdown()
        server.server_close()

    assert len(_FakeLMStudioHandler.payloads) == 2
    assert _FakeLMStudioHandler.payloads[0]["model"] == "deep-model"
    assert _FakeLMStudioHandler.payloads[0].get("tools")
    assert _FakeLMStudioHandler.payloads[1]["model"] == "deep-model"
    assert "tools" not in _FakeLMStudioHandler.payloads[1]


def _start_fake_lmstudio(
    delay_seconds: float = 0.0,
    reject_tools: bool = False,
) -> ThreadingHTTPServer:
    _FakeLMStudioHandler.delay_seconds = delay_seconds
    _FakeLMStudioHandler.reject_tools = reject_tools
    _FakeLMStudioHandler.active = 0
    _FakeLMStudioHandler.max_active = 0
    _FakeLMStudioHandler.payloads = []

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLMStudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _lmstudio_config(server: ThreadingHTTPServer) -> Config:
    config = Config()
    config.llm_enabled = True
    config.llm_provider = "lmstudio"
    config.llm_model = "base-model"
    config.llm_chat_model = "chat-model"
    config.llm_deep_model = "deep-model"
    config.llm_base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    config.llm_max_attempts = 1
    config.llm_max_tokens = 20
    config.llm_timeout_seconds = 5
    config.llm_chat_timeout_seconds = 5
    config.llm_reasoning_effort = "low"
    return config
