"""Optional local LLM adapter for Aeon-V1 Layer 4.

No provider is a hard dependency. generate_text() returns None whenever LLM
is disabled or any error occurs — callers always fall back to rule-based behavior.

Provider:
    lmstudio    — LM Studio local server (OpenAI-compatible REST, no extra packages)

Environment variables:
    AEON_V1_LLM=1                        — enable LLM
    AEON_V1_LLM_MODEL=google/gemma-4-e4b — local model name
    AEON_V1_LLM_SEARCH_MODEL=mistral/...  — optional memory-search planner model
    AEON_V1_LLM_BASE_URL=http://...      — LM Studio base URL (default: http://localhost:1234/v1)
    AEON_V1_LLM_REASONING_EFFORT=low     — LM Studio reasoning effort for reasoning models
"""
import json
import os
import re
import threading
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .time_utils import utc_now_iso

# Hard cap: at most 10 concurrent HTTP requests to LM Studio.
# This governs external HTTP concurrency, not inter-agent messaging.
# Inter-agent communication goes through the message bus (bus.py).
_LM_STUDIO_MAX_QUEUE = 10
_lm_studio_semaphore = threading.BoundedSemaphore(_LM_STUDIO_MAX_QUEUE)

# Tool definition passed to the LLM so it can query the memory index agent.
QUERY_MEMORY_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": (
            "Search the memory store for relevant memories. "
            "Call this before writing your response to retrieve the context you need."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "reflections"],
                    },
                    "description": "Memory types to search. Omit to search all types.",
                },
            },
            "required": ["query"],
        },
    },
}


def generate_text(prompt: str, config: Optional[Config] = None) -> Optional[str]:
    """Call the configured LLM and return the response text, or None on any failure.

    Args:
        prompt: The full prompt to send.
        config: Aeon-V1 Config — defaults to Config() if None.

    Returns:
        Response string, or None if LLM is disabled / unavailable / errored.
    """
    if config is None:
        config = Config()
    if not config.llm_enabled:
        return None
    return _call_lmstudio_messages([{"role": "user", "content": prompt}], config)


def generate_chat(messages: List[Dict], config: Optional[Config] = None) -> Optional[str]:
    """Call the configured chat LLM with explicit role-separated messages."""
    if config is None:
        config = Config()
    if not config.llm_enabled:
        return None
    result = _call_lmstudio_messages(messages, config, model=config.llm_chat_model)
    if result:
        return result
    if config.llm_chat_model != config.llm_model:
        return _call_lmstudio_messages(messages, config, model=config.llm_model)
    return None


def generate_search_text(prompt: str, config: Optional[Config] = None) -> Optional[str]:
    """Call the configured memory-search planner model.

    This role is intentionally separate from chat. It can use a model like
    Mistral to turn a fuzzy recall request into concrete search queries without
    changing Aeon's conversational model.
    """
    if config is None:
        config = Config()
    if not config.llm_enabled:
        return None
    return _call_lmstudio_messages(
        [{"role": "user", "content": prompt}],
        config,
        model=config.llm_search_model,
        timeout=config.llm_search_timeout_seconds,
        include_reasoning=False,
    )


def generate_music_chat(messages: List[Dict], config: Optional[Config] = None) -> Optional[str]:
    """Call Aeon's machine-local music worker through its compatible API."""
    if config is None:
        config = Config()
    if not config.llm_enabled or not config.llm_music_model:
        return None
    return _call_lmstudio_messages(
        messages,
        config,
        model=config.llm_music_model,
        timeout=config.llm_music_timeout_seconds,
        include_reasoning=False,
        base_url=config.llm_music_base_url,
    )


def generate_image_description(
    image_path: Path,
    prompt: str,
    config: Optional[Config] = None,
) -> Optional[str]:
    """Describe an image through an OpenAI-compatible vision model."""
    if config is None:
        config = Config()
    if not config.llm_enabled:
        return None
    model = resolve_lmstudio_vision_model(config)
    try:
        mime = _image_mime_type(image_path)
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except Exception:
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                },
            ],
        }
    ]
    return _call_lmstudio_messages(
        messages,
        config,
        model=model,
        timeout=config.llm_media_timeout_seconds,
        include_reasoning=False,
    ) or _call_lmstudio_native_image(
        data_url=f"data:{mime};base64,{data}",
        prompt=prompt,
        config=config,
        model=model,
    )



_LM_STUDIO_MAX_ATTEMPTS = 5


def _call_lmstudio(prompt: str, config: Config) -> Optional[str]:
    """Call LM Studio local server via OpenAI-compatible REST API. No extra packages needed.

    Retries up to _LM_STUDIO_MAX_ATTEMPTS times on failure, then returns None.
    """
    return _call_lmstudio_messages([{"role": "user", "content": prompt}], config)


def _call_lmstudio_messages(
    messages: List[Dict],
    config: Config,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    include_reasoning: bool = True,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """Call LM Studio local server with OpenAI-compatible chat messages."""
    url = f"{(base_url or config.llm_base_url).rstrip('/')}/chat/completions"
    if not _lm_studio_semaphore.acquire(blocking=False):
        return None  # queue full — 10 requests already in flight

    try:
        max_attempts = max(1, config.llm_max_attempts)
        for attempt in range(1, max_attempts + 1):
            payload_data = {
                "model": model or config.llm_model,
                "messages": messages,
                "temperature": config.llm_temperature,
                "max_tokens": min(config.llm_max_tokens * attempt, 1024),
            }
            if include_reasoning and config.llm_reasoning_effort:
                payload_data["reasoning_effort"] = config.llm_reasoning_effort
            payload = json.dumps(payload_data).encode("utf-8")
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout or config.llm_chat_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    message = data["choices"][0]["message"]
                    content = (message.get("content") or "").strip()
                    if content:
                        return content
            except Exception:
                if attempt == max_attempts:
                    return None
        return None
    finally:
        _lm_studio_semaphore.release()


def _call_lmstudio_native_image(
    data_url: str,
    prompt: str,
    config: Config,
    model: str,
) -> Optional[str]:
    """Fallback to LM Studio's native /api/v1/chat image input shape."""
    base_url = config.llm_base_url.rstrip("/")
    if base_url.endswith("/v1"):
        native_base = base_url[:-3]
    else:
        native_base = base_url
    url = f"{native_base}/api/v1/chat"

    payload_data = {
        "model": model,
        "input": [
            {"type": "text", "content": prompt},
            {"type": "image", "data_url": data_url},
        ],
        "temperature": config.llm_temperature,
        "max_output_tokens": min(config.llm_max_tokens, 1024),
        "store": False,
    }
    payload = json.dumps(payload_data).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.llm_media_timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    for item in data.get("output", []):
        if item.get("type") == "message":
            content = (item.get("content") or "").strip()
            if content:
                return content
    return None


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _detect_lmstudio_vision_model(config: Config) -> str:
    """Pick a likely loaded vision model from LM Studio when no env var is set."""
    url = f"{config.llm_base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    candidates = []
    for item in data.get("data", []):
        model_id = str(item.get("id", ""))
        lowered = model_id.lower()
        if any(token in lowered for token in ("vl", "vision", "visual", "llava", "minicpm-v", "gemma-3")):
            candidates.append(model_id)
    return candidates[0] if candidates else ""


def resolve_lmstudio_vision_model(config: Config) -> str:
    """Resolve the model Aeon should use for image analysis."""
    return (
        config.llm_vision_model
        or _detect_lmstudio_vision_model(config)
        or config.llm_chat_model
        or config.llm_model
    )


def generate_with_memory(
    prompt: str,
    index_agent,
    config: Optional[Config] = None,
) -> Optional[str]:
    """Call the LLM with query_memory tool access.

    The LLM may call query_memory up to _LM_STUDIO_MAX_ATTEMPTS times to
    fetch relevant memories before producing a final text response.
    Falls back to an inlined local LM Studio prompt when tool calling is off.

    All communication between the LLM loop and index_agent goes through the
    message bus — index_agent._handle_bus_query is registered transiently for
    the duration of this call and removed when it returns.

    Args:
        prompt:      The task prompt (no memory content inlined).
        index_agent: MemoryIndexAgent instance whose bus handler is used.
        config:      Aeon-V1 Config.

    Returns:
        Final LLM response string, or None on failure.
    """
    if config is None:
        config = Config()
    if not config.llm_enabled:
        return None

    from .bus import get_bus
    bus = get_bus()
    bus.subscribe("memory.query", index_agent._handle_bus_query)
    try:
        result = _call_lmstudio_with_tools(prompt, config)
        if result:
            return result
        result = _call_lmstudio_messages(
            [{"role": "user", "content": prompt}],
            config,
            model=config.llm_deep_model,
        )
        if result:
            return result
        if config.llm_deep_model != config.llm_model:
            return _call_lmstudio_messages(
                [{"role": "user", "content": prompt}],
                config,
                model=config.llm_model,
            )
        return None
    finally:
        bus.unsubscribe("memory.query", index_agent._handle_bus_query)


def _call_lmstudio_with_tools(prompt: str, config: Config) -> Optional[str]:
    """Tool-calling loop for LM Studio: LLM queries memory agent via bus, then responds."""
    if not _lm_studio_semaphore.acquire(blocking=False):
        return None

    from .bus import get_bus
    from .schemas import make_agent_message

    url = f"{config.llm_base_url.rstrip('/')}/chat/completions"
    messages: List[Dict] = [{"role": "user", "content": prompt}]

    try:
        for _ in range(_LM_STUDIO_MAX_ATTEMPTS):
            payload_data = {
                "model":       config.llm_deep_model,
                "messages":    messages,
                "tools":       [QUERY_MEMORY_TOOL],
                "tool_choice": "auto",
                "temperature": config.llm_temperature,
                "max_tokens":  config.llm_max_tokens,
            }
            if config.llm_reasoning_effort:
                payload_data["reasoning_effort"] = config.llm_reasoning_effort
            payload = json.dumps(payload_data).encode("utf-8")
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=config.llm_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception:
                return None

            choice = data["choices"][0]
            message = choice["message"]

            if choice.get("finish_reason") == "tool_calls" or message.get("tool_calls"):
                messages.append(message)
                for tc in message.get("tool_calls", []):
                    bus_msg = make_agent_message(
                        agent_id="llm",
                        action="read",
                        target="memory_index",
                        payload={
                            "name":      tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                        status="pending",
                        timestamp=utc_now_iso(),
                        requires_approval=False,
                    )
                    result = get_bus().request("memory.query", bus_msg)
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      result or "{}",
                    })
                continue  # send tool results back to LLM

            content = message.get("content", "")
            return content if content else None

        return None  # loop limit reached
    finally:
        _lm_studio_semaphore.release()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_reflection_prompt(analysis: Dict) -> str:
    """Build a compact prompt for LLM-enhanced reflection narrative sections.

    The LLM is asked to write ONLY 4 sections using provided data.
    Safety rules are embedded in the prompt.
    """
    ep_count = analysis["source_types"].get("episodic", 0)
    sem_count = analysis["source_types"].get("semantic", 0)

    mem_lines: List[str] = []
    for m in analysis.get("sources", [])[:10]:
        if m["type"] == "episodic":
            text = m.get("summary", "")[:120]
        else:
            concept = m.get("concept", "")
            desc = m.get("description", "")[:100]
            text = f"{concept}: {desc}" if concept else desc
        mem_lines.append(
            f"- [{m['type']}] importance={m.get('importance', 0):.2f}: {text}"
        )

    patterns_text = "\n".join(f"- {p}" for p in analysis.get("detected_patterns", []))
    uncertainty_text = "\n".join(f"- {u}" for u in analysis.get("uncertainty_notes", []))
    tasks_text = "\n".join(f"- {t}" for t in analysis.get("suggested_tasks", []))

    return f"""You are assisting an AI memory system with reflection synthesis.

CONTEXT:
- {ep_count} episodic and {sem_count} semantic memories reviewed.
- Confidence: {analysis.get('confidence', 0):.2f}

MEMORIES (up to 10):
{chr(10).join(mem_lines) or "- None"}

RAW PATTERN SIGNALS:
{patterns_text or "- None detected"}

RAW UNCERTAINTY SIGNALS:
{uncertainty_text or "- None detected"}

RAW TASK SIGNALS:
{tasks_text or "- None detected"}

TASK:
Write exactly 4 reflection sections using ONLY the data above.
Keep each section to 3-6 bullet points. Be specific and concise.

SAFETY RULES (mandatory):
- Use only information provided — do not invent facts, events, or outcomes.
- Do not suggest shell commands, system execution, or deployment actions.
- Do not claim any action was taken or completed.
- Core memory changes are SUGGESTIONS ONLY — humans decide what enters vault/core/.
- Do not alter source IDs, tags, or stored metadata.

OUTPUT FORMAT — use exactly these headers in this order:

### What Was Learned
[bullet points from high-importance memories]

### New Patterns Noticed
[bullet points about recurring themes or trends]

### Conflicts or Uncertainty
[bullet points about unclear or conflicting information]

### Suggested Tasks
[bullet points about implied next steps to investigate]

Write only these 4 sections. Nothing before or after."""


def build_simulation_prompt(task: Dict) -> str:
    """Build a compact prompt for LLM-enhanced simulation planning.

    The LLM proposes an action plan. Safety constraints are embedded.
    """
    return f"""You are assisting an AI memory system with action simulation planning.

TASK:
Title: {task.get('title', '')}
Description: {task.get('description', '')}
Priority: {task.get('priority', 0.5)}
Confidence: {task.get('confidence', 0.5)}

TASK:
Analyze this task and write a simulation plan. Be specific but concise.

SAFETY RULES (mandatory):
- This is SIMULATION ONLY — no real commands will be executed.
- Do not suggest subprocess calls, shell commands, or direct system actions.
- All proposed actions require explicit human approval before any execution.
- Do not claim actions were completed. Describe what WOULD happen.
- Keep the plan grounded in what is described — do not invent requirements.

OUTPUT FORMAT — use exactly these headers in this order:

### Proposed Action
[1-2 sentences: what should happen, concretely]

### Expected Outcome
[1-2 sentences: the realistic result if the action succeeds]

### Risk Assessment
[2-4 bullet points: risks and required approvals]

Write only these 3 sections. Nothing before or after."""


# ---------------------------------------------------------------------------
# Sparse prompt builders (for tool-calling path — no memory inlined)
# ---------------------------------------------------------------------------

def build_reflection_prompt_sparse(analysis: Dict) -> str:
    """Reflection prompt for the tool-calling path.

    Does NOT inline memory content — the LLM queries the index agent instead.
    """
    ep_count = analysis["source_types"].get("episodic", 0)
    sem_count = analysis["source_types"].get("semantic", 0)
    patterns_text  = "\n".join(f"- {p}" for p in analysis.get("detected_patterns", []))
    uncertainty_text = "\n".join(f"- {u}" for u in analysis.get("uncertainty_notes", []))

    return f"""You are assisting an AI memory system with reflection synthesis.

CONTEXT:
- {ep_count} episodic and {sem_count} semantic memories are available in the store.
- Confidence score: {analysis.get('confidence', 0):.2f}

RAW PATTERN SIGNALS:
{patterns_text or "- None detected"}

RAW UNCERTAINTY SIGNALS:
{uncertainty_text or "- None detected"}

TASK:
Use the query_memory tool to retrieve relevant memories (1-3 targeted queries), then write exactly 4 reflection sections.
Keep each section to 3-6 bullet points. Be specific and concise.

SAFETY RULES (mandatory):
- Use only information returned by query_memory — do not invent facts, events, or outcomes.
- Do not suggest shell commands, system execution, or deployment actions.
- Do not claim any action was taken or completed.
- Core memory changes are SUGGESTIONS ONLY — humans decide what enters vault/core/.
- Do not alter source IDs, tags, or stored metadata.

OUTPUT FORMAT — use exactly these headers in this order:

### What Was Learned
[bullet points from high-importance memories]

### New Patterns Noticed
[bullet points about recurring themes or trends]

### Conflicts or Uncertainty
[bullet points about unclear or conflicting information]

### Suggested Tasks
[bullet points about implied next steps to investigate]

Write only these 4 sections. Nothing before or after."""


def build_conversation_arc_prompt(arc: Dict) -> str:
    """Build a prompt for LLM-enhanced conversation arc narrative.

    The LLM is given the raw arc data and asked to write a concise episodic
    summary. Falls back to rule-based text if LLM is unavailable.
    """
    shifts_text = "\n".join(
        f"- Turn {s['at_turn']}: {s['from_intent']} → {s['to_intent']}"
        + (f" [{s['speaker']}]: \"{s['snippet']}\"" if s.get("snippet") else "")
        for s in arc.get("shifts", [])
    ) or "- None detected"

    return f"""You are summarizing a conversation arc for an AI memory system.

CONVERSATION DATA:
- Session: {arc.get('session_id', '')}
- Turns: {arc.get('turn_count', 0)}
- Dominant intent: {arc.get('dominant_intent', 'general')}
- Intent shifts detected:
{shifts_text}

TASK:
Write a single concise paragraph (3-5 sentences) that describes how the user's \
needs or focus evolved during this conversation. Highlight any noteworthy shifts \
in tone or intent. Use first-person from the memory system's perspective \
(e.g. "I noticed the user shifted from...").

SAFETY RULES:
- Use only the data provided — do not invent quotes or events.
- Do not suggest actions or commands.
- Do not reference specific usernames or private information.

Write only the paragraph. Nothing before or after."""


def score_importance(text: str, config: Optional[Config] = None) -> Optional[float]:
    """Ask the LLM to rate memory importance on [0.0, 1.0].

    Uses the search/background model so the main chat model is not interrupted.
    Returns None on failure so callers can use rule-based scoring without crashing.
    """
    if config is None:
        config = Config()
    if not config.llm_enabled:
        return None

    prompt = (
        "Rate the long-term importance of the following memory on a scale of 0.0 to 1.0.\n\n"
        "0.0 = trivial or ephemeral (casual chat, already-resolved status, filler)\n"
        "0.5 = useful context (technique, minor insight, situational knowledge)\n"
        "1.0 = highly significant (key decision, major insight, critical system behavior)\n\n"
        "Respond with only a single decimal number between 0.0 and 1.0. "
        "No explanation, no units, no extra text.\n\n"
        f"Memory:\n{text[:800]}"
    )
    raw = generate_search_text(prompt, config)
    if not raw:
        return None
    match = re.search(r"\b(0(\.\d+)?|1(\.0*)?)\b", raw.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(0))))
    return None


def build_simulation_prompt_sparse(task: Dict) -> str:
    """Simulation prompt for the tool-calling path — no context inlined."""
    return f"""You are assisting an AI memory system with action simulation planning.

TASK:
Title: {task.get('title', '')}
Description: {task.get('description', '')}
Priority: {task.get('priority', 0.5)}
Confidence: {task.get('confidence', 0.5)}

Use the query_memory tool to retrieve relevant context for this task, then write a simulation plan.

SAFETY RULES (mandatory):
- This is SIMULATION ONLY — no real commands will be executed.
- Do not suggest subprocess calls, shell commands, or direct system actions.
- All proposed actions require explicit human approval before any execution.
- Do not claim actions were completed. Describe what WOULD happen.
- Keep the plan grounded in what is described — do not invent requirements.

OUTPUT FORMAT — use exactly these headers in this order:

### Proposed Action
[1-2 sentences: what should happen, concretely]

### Expected Outcome
[1-2 sentences: the realistic result if the action succeeds]

### Risk Assessment
[2-4 bullet points: risks and required approvals]

Write only these 3 sections. Nothing before or after."""


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_reflection_sections(text: str) -> Dict[str, str]:
    """Extract the 4 narrative sections from an LLM reflection response.

    Returns a dict of {section_name: content}. Missing sections are omitted.
    """
    section_names = [
        "What Was Learned",
        "New Patterns Noticed",
        "Conflicts or Uncertainty",
        "Suggested Tasks",
    ]
    return _extract_sections(text, section_names)


def parse_simulation_sections(text: str) -> Dict[str, str]:
    """Extract the 3 simulation sections from an LLM simulation response."""
    return _extract_sections(text, ["Proposed Action", "Expected Outcome", "Risk Assessment"])


def _extract_sections(text: str, names: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name in names:
        m = re.search(
            rf"###\s*{re.escape(name)}\s*\n(.*?)(?=###\s|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            content = m.group(1).strip()
            if content:
                result[name] = content
    return result
