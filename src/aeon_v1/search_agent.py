"""SearchAgent — natural-language memory search for the dashboard chat.

Sits on top of search.py (which Codex is patching separately).
Detects search intent in chat messages, extracts the topic, runs a
multi-keyword fan-out search, and returns a human-readable chat reply.

Bus topic  : search.query
  payload  : {query, memory_types (optional), limit (optional)}
  returns  : JSON string — serialised list of formatted result dicts

Codex connection point
----------------------
Add to DashboardController.__init__() in dashboard.py:

    from .search_agent import SearchAgent
    self._search_agent = SearchAgent(self.config)

Add to DashboardController.chat() before build_response():

    search_reply = self._search_agent.handle_chat_query(effective_text)
    if search_reply is not None:
        response = search_reply
    else:
        response = build_response(
            user_text=effective_text,
            memories=memories,
            history=self._chat_history[-4:],
            config=self.config,
            index_agent=self._index_agent,
        )
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .config import Config
from .llm import generate_search_text
from .search import search


# ------------------------------------------------------------------ intent patterns

# Each pattern must capture the topic substring in group 1.
# Ordered from most-specific to least-specific so the first match wins cleanly.
_INTENT_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"what\s+(?:number|code|phrase|fact)\s+(?:are\s+you\s+supposed\s+to\s+remember|did\s+i\s+ask\s+you\s+to\s+remember|was\s+the\s+test|was\s+the\s+memory\s+test)\??",
        r"(?:i\s+told\s+you|i\s+gave\s+you|there\s+was)\s+(?:a\s+)?(?:specific\s+)?(?:six[-\s]?digit|6[-\s]?digit|random|test)?\s*(?:number|code|phrase|fact).*(?:remember|memory|test)",
        r"(?:six[-\s]?digit|6[-\s]?digit)\s+number",
        r"what\s+(?:do\s+(?:you|i)\s+(?:know|remember|recall|have)|(?:have\s+(?:you|i)\s+(?:learned|stored|saved|noted)))\s+(?:about|on|regarding|of)\s+(.+)",
        r"do\s+(?:you|i)\s+(?:know|remember|recall|have)\s+(?:anything\s+)?(?:about|on|regarding|of)\s+(.+)",
        r"what(?:'s|\s+is|\s+was)\s+(?:stored|saved|noted|in\s+memory)\s+(?:about|on|regarding|for)\s+(.+)",
        r"what\s+(?:memories?|notes?)\s+do\s+(?:you|i)\s+have\s+(?:on|about|regarding|of)\s+(.+)",
        r"(?:any|what)\s+(?:memories?|notes?|info(?:rmation)?)\s+(?:about|on|regarding|of)\s+(.+)",
        r"tell\s+me\s+(?:everything\s+)?(?:you\s+)?(?:know\s+)?about\s+(.+)",
        r"recall\s+(?:everything\s+about\s+|anything\s+about\s+|what\s+(?:you|i)\s+(?:know|have)\s+about\s+)?(.+)",
        r"search\s+(?:(?:my\s+)?memories?\s+)?(?:for\s+)?(?:(?:about|on|for|regarding|related\s+to)\s+)?(.+)",
        r"(?:look\s+up|lookup)\s+(.+?)(?:\s+(?:in|from)\s+(?:my\s+)?mem(?:ory|ories))?$",
        r"(?:find|list|show)\s+(?:me\s+)?(?:my\s+)?memories?\s+(?:about|on|for|regarding|of)\s+(.+)",
        r"(?:find|show)\s+(?:me\s+)?(?:everything\s+)?(?:about|on|for|regarding)\s+(.+)",
        r"memories?\s+(?:about|on|for|regarding|of)\s+(.+)",
    ]
]

# Words stripped from an extracted topic before tokenisation
_TOPIC_STOP = {
    "please", "thanks", "thank", "you", "me", "my", "our", "i", "we",
    "it", "this", "that", "more", "some", "any", "all", "now", "here",
    "there", "very", "really",
}

_TOPIC_ALIASES = {
    "aeon": "aeon local recursive memory dashboard obsidian lm studio",
    "you": "aeon local recursive memory dashboard obsidian lm studio",
    "yourself": "aeon local recursive memory dashboard obsidian lm studio",
    "your memory": "aeon local recursive memory dashboard obsidian lm studio",
}

# Maximum number of individual keywords to fan-out per search
_MAX_KEYWORDS = 6

# Minimum character length for a keyword to be worth searching
_MIN_KW_LEN = 3

_KEYWORD_STOP = _TOPIC_STOP | {
    "memory", "memories", "remember", "recall", "stored", "saved", "local",
}

_NUMERIC_RECALL_TOPIC = "six digit number memory test"

# Field read-order when extracting a human-readable snippet from a memory record
_SNIPPET_FIELDS = ("summary", "description", "concept", "content", "text", "title")

# Maximum characters shown per result snippet in the chat reply
_SNIPPET_MAX = 220


# ------------------------------------------------------------------ public class

class SearchAgent:
    """Handles natural-language memory search queries from the dashboard chat.

    Direct usage (no bus required):
        agent = SearchAgent(config)
        reply = agent.handle_chat_query("what do you know about Python?")
        # Returns a formatted string, or None if the text is not a search query.

    Bus usage:
        bus.subscribe("search.query", agent._handle_bus_query)
        result_json = bus.request("search.query", msg)  # returns JSON string
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    # -------------------------------------------------------------- public API

    def is_search_query(self, text: str) -> bool:
        """Return True when *text* looks like a memory-search request."""
        return self._extract_topic(text.strip()) is not None

    def handle_chat_query(self, text: str) -> Optional[str]:
        """Entry point for the dashboard chat flow.

        Returns a formatted reply string when *text* is a search query,
        or None when it is ordinary conversation (caller falls through to
        the normal LLM / local-fallback path).
        """
        reply = self.handle_chat_query_with_ids(text)
        return None if reply is None else str(reply["reply"])

    def handle_chat_query_with_ids(self, text: str) -> Optional[Dict]:
        """Return a search reply plus the source memory IDs it used."""
        topic = self._extract_topic(text.strip())
        if topic is None:
            return None
        expanded_topic = self._expand_topic(topic)
        results = self.results(expanded_topic)
        reply = (
            self._format_numeric_reply(results, topic)
            if self._is_numeric_recall_query(expanded_topic)
            else self._format_reply(results, topic)
        )
        return {
            "reply": reply,
            "memory_ids": [
                r.get("memory", {}).get("id", "")
                for r in results
                if r.get("memory", {}).get("id")
            ],
        }

    def query(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Run a search and return a list of formatted result dicts.

        For programmatic callers that need raw data rather than a chat reply.
        """
        results = self._fan_out_search(query, memory_types=memory_types, limit=limit)
        return [self._result_dict(r) for r in results]

    def results(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Run a natural-language search and return raw search result records.

        Chat prompt builders need the original memory dicts so they can preserve
        source IDs, memory types, and readable snippets.
        """
        return self._fan_out_search(query, memory_types=memory_types, limit=limit)

    # ------------------------------------------------------------ bus handler

    def _handle_bus_query(self, message: Dict) -> str:
        """Bus entry point — subscribe to 'search.query' on the MessageBus.

        Payload: {query, memory_types (optional), limit (optional)}
        Returns:  JSON string — list of formatted result dicts.
        """
        payload = message.get("payload", {})
        results = self.query(
            query=payload.get("query", ""),
            memory_types=payload.get("memory_types"),
            limit=int(payload.get("limit", 10)),
        )
        return json.dumps(results, ensure_ascii=False)

    # --------------------------------------------------------------- internals

    def _extract_topic(self, text: str) -> Optional[str]:
        """Match *text* against intent patterns and return the cleaned topic, or None."""
        for pattern in _INTENT_PATTERNS:
            m = pattern.search(text)
            if m:
                raw = (m.group(1) if m.groups() else _NUMERIC_RECALL_TOPIC).strip(" ?.!,;:")
                cleaned = self._clean_topic(raw)
                if cleaned:
                    return cleaned
        return None

    def _clean_topic(self, topic: str) -> str:
        """Strip trailing punctuation and low-signal words from a raw topic."""
        topic = topic.strip(" ?.!,;:")
        tokens = [t for t in topic.split() if t.lower() not in _TOPIC_STOP]
        cleaned = " ".join(tokens).strip(" ?.!,;:")
        return cleaned

    def _expand_topic(self, topic: str) -> str:
        """Expand broad self-references into concrete Aeon project terms."""
        return _TOPIC_ALIASES.get(topic.lower(), topic)

    def _keywords(self, topic: str) -> List[str]:
        """Build an ordered, deduplicated keyword list from a topic string.

        The full phrase always comes first (so an exact phrase hit ranks highest
        in search.py), followed by individual tokens long enough to be meaningful.
        """
        keywords: List[str] = []
        phrase = topic.strip()
        if phrase:
            keywords.append(phrase)

        for word in re.split(r"[\s_\-/]+", phrase):
            token = re.sub(r"[^a-z0-9]", "", word.lower())
            if len(token) >= _MIN_KW_LEN and token not in _KEYWORD_STOP and token not in keywords:
                keywords.append(token)

        return keywords[:_MAX_KEYWORDS]

    def _fan_out_search(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search each keyword independently and merge results, deduplicating by ID."""
        if memory_types is None:
            memory_types = [
                "semantic", "episodic", "consolidations", "reflections", "raw",
            ]
            if self._should_search_media(query):
                memory_types.append("media")

        if self._is_numeric_recall_query(query):
            numeric = self._numeric_recall_results(memory_types, limit)
            if numeric:
                return numeric

        seen_ids: set = set()
        merged: List[Dict] = []

        for keyword in self._planned_keywords(query):
            for result in search(keyword, memory_types=memory_types, config=self.config):
                mem = result.get("memory", {})
                mem_id = mem.get("id", "")
                snippet = self._snippet(mem)
                if (
                    not mem_id
                    or mem_id in seen_ids
                    or not snippet
                    or self._is_search_echo(snippet)
                    or self._is_query_echo(snippet)
                ):
                    continue
                seen_ids.add(mem_id)
                merged.append(result)

        merged.sort(
            key=lambda r: (
                float(r.get("score", 0) or 0),
                float(r.get("memory", {}).get("importance", 0) or 0),
            ),
            reverse=True,
        )
        return self._diversify(merged, limit)

    def _planned_keywords(self, query: str) -> List[str]:
        """Use the search model role to plan queries, then add local fallback terms."""
        keywords: List[str] = []
        for keyword in self._llm_keywords(query) + self._keywords(query):
            cleaned = keyword.strip()
            if cleaned and cleaned.lower() not in [k.lower() for k in keywords]:
                keywords.append(cleaned)
        return keywords[:10]

    def _llm_keywords(self, query: str) -> List[str]:
        prompt = f"""You are Aeon's memory search planner.

Turn the user's recall request into concrete search queries for local JSON/Markdown memories.
Return ONLY valid JSON in this shape:
{{"queries":["short exact query 1","short exact query 2"]}}

Rules:
- Include exact IDs, numbers, filenames, quoted phrases, or odd tokens from the request.
- For fuzzy factual recall, include likely stored wording variants.
- For number recall, include "six digit number", "random number", "test number", and "asked me to remember" when relevant.
- Do not answer the user. Only produce search queries.
- Maximum 6 queries.

User request: {query}
"""
        text = generate_search_text(prompt, self.config)
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return []
            try:
                payload = json.loads(match.group(0))
            except Exception:
                return []
        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            return []
        return [str(q).strip() for q in queries if str(q).strip()][:6]

    def _diversify(self, results: List[Dict], limit: int) -> List[Dict]:
        """Favor a spread of memory layers before filling by rank."""
        preferred_types = ["semantic", "episodic", "consolidations", "reflections", "media", "raw"]
        selected: List[Dict] = []
        seen_ids: set = set()

        for memory_type in preferred_types:
            for result in results:
                mem = result.get("memory", {})
                mem_id = mem.get("id", "")
                result_type = mem.get("type") or str(result.get("match_type", "")).split("/")[-1]
                if result_type == memory_type and mem_id and mem_id not in seen_ids:
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

    def _is_numeric_recall_query(self, query: str) -> bool:
        lowered = query.lower()
        has_number_word = any(token in lowered for token in ("number", "six digit", "six-digit", "6 digit", "6-digit", "digits", "code"))
        has_memory_word = any(token in lowered for token in ("remember", "memory", "test", "asked", "supposed", "specific", "random"))
        return has_number_word and has_memory_word

    def _numeric_recall_results(self, memory_types: List[str], limit: int) -> List[Dict]:
        """Find stored six-digit facts directly instead of relying on broad keywords."""
        results: List[Dict] = []
        seen_ids: set = set()
        for memory_type in memory_types:
            mem_dir = self.config.memory_path / memory_type
            if not mem_dir.exists():
                continue
            for path in mem_dir.glob("*.json"):
                try:
                    mem = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                snippet = self._snippet(mem)
                if not snippet or self._is_search_echo(snippet) or self._is_query_echo(snippet):
                    continue
                if not re.search(r"\b\d{6}\b", snippet):
                    continue
                mem_id = mem.get("id", "")
                if not mem_id or mem_id in seen_ids:
                    continue
                seen_ids.add(mem_id)
                lowered = snippet.lower()
                score = 20.0
                score += sum(
                    2.0
                    for token in ("remember", "asked", "test", "random", "specific", "important")
                    if token in lowered
                )
                results.append({
                    "match_type": memory_type,
                    "file": str(path),
                    "memory": mem,
                    "score": score,
                })

        results.sort(
            key=lambda r: (
                float(r.get("score", 0) or 0),
                float(r.get("memory", {}).get("importance", 0) or 0),
            ),
            reverse=True,
        )
        return self._diversify(results, limit)

    def _should_search_media(self, query: str) -> bool:
        lowered = query.lower()
        return any(
            token in lowered
            for token in ("image", "photo", "picture", "screenshot", "video", "audio", "sound", "media")
        )

    def _snippet(self, mem: Dict) -> str:
        """Extract the most readable content snippet from a memory dict."""
        for field in _SNIPPET_FIELDS:
            val = mem.get(field)
            if val and str(val).strip():
                text = str(val).strip()
                text = re.sub(r"^\[?(raw|episodic|semantic|reflection|reflections|media)\]?\s+", "", text, flags=re.IGNORECASE)
                if len(text) > _SNIPPET_MAX:
                    text = text[: _SNIPPET_MAX - 3].rstrip() + "..."
                return text
        return ""

    def _is_search_echo(self, snippet: str) -> bool:
        """Avoid recursively returning prior search-agent replies as memories."""
        normalized = re.sub(r"^(user|aeon):\s*", "", snippet.strip(), flags=re.IGNORECASE)
        return normalized.lower().startswith("here is what i have in memory about")

    def _is_query_echo(self, snippet: str) -> bool:
        """Avoid returning prior memory-search questions as answers."""
        normalized = re.sub(r"^(user|aeon):\s*", "", snippet.strip(), flags=re.IGNORECASE)
        lowered = normalized.lower()
        question_leads = (
            "what ", "do ", "any ", "search ", "look up", "lookup ",
            "find ", "list ", "show ", "memories ", "tell me", "recall ",
        )
        return lowered.startswith(question_leads) and self.is_search_query(normalized)

    def _result_dict(self, result: Dict) -> Dict:
        """Reduce a raw search result to a clean summary dict for programmatic use."""
        mem = result.get("memory", {})
        entry: Dict = {
            "id": mem.get("id", ""),
            "type": mem.get("type") or str(result.get("match_type", "")).split("/")[-1],
            "importance": float(mem.get("importance", 0) or 0),
            "score": float(result.get("score", 0) or 0),
        }
        snippet = self._snippet(mem)
        if snippet:
            entry["content"] = snippet
        return entry

    def _format_reply(self, results: List[Dict], topic: str) -> str:
        """Build a human-readable chat reply from fan-out search results."""
        if not results:
            return (
                f"I searched my memory for '{topic}' but did not find anything "
                "stored about that yet."
            )

        lines: List[str] = [f"Here is what I have in memory about '{topic}':"]
        shown = 0
        for result in results:
            mem = result.get("memory", {})
            mem_type = (
                mem.get("type")
                or str(result.get("match_type", "memory")).split("/")[-1]
            )
            snippet = self._snippet(mem)
            if not snippet:
                continue
            shown += 1
            lines.append(f"{shown}. [{mem_type}] {snippet}")
            if shown >= 8:
                break

        if shown == 0:
            return (
                f"I found memory entries matching '{topic}' but they had "
                "no readable content attached."
            )

        return "\n".join(lines)

    def _format_numeric_reply(self, results: List[Dict], topic: str) -> str:
        """Answer numeric recall requests directly before showing evidence."""
        if not results:
            return (
                f"I searched my memory for '{topic}' but did not find a stored "
                "six-digit number for that request."
            )

        counts: Dict[str, int] = {}
        for result in results:
            snippet = self._snippet(result.get("memory", {}))
            for number in re.findall(r"\b\d{6}\b", snippet):
                counts[number] = counts.get(number, 0) + 1

        if not counts:
            return self._format_reply(results, topic)

        number = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
        lines = [f"The number is {number}."]
        lines.append("I found it in local memory, not by guessing.")
        shown = 0
        for result in results:
            mem = result.get("memory", {})
            snippet = self._snippet(mem)
            if number not in snippet:
                continue
            mem_type = mem.get("type") or str(result.get("match_type", "memory")).split("/")[-1]
            mem_id = mem.get("id", "unknown")
            shown += 1
            lines.append(f"Source {shown}: [{mem_type}] {mem_id}: {snippet}")
            if shown >= 3:
                break
        return "\n".join(lines)
