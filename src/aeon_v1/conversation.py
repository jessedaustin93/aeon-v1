"""Conversation arc tracking for Aeon-V1.

Tracks turn-by-turn intent within a session and detects shifts — e.g. when a
request moves from task-focused to reflective. At session end, close() ingests
the arc as a regular episodic memory so it surfaces naturally in reflection.

This module is purely additive: nothing in ingest, reflect, or memory_store
is modified. Existing callers are unaffected.
"""
from typing import Dict, List, Optional, Tuple

from .config import Config

# ---------------------------------------------------------------------------
# Intent classification signals
# ---------------------------------------------------------------------------

_INTENT_SIGNALS: Dict[str, List[str]] = {
    "task":        ["need to", "should ", "build ", "fix ", "implement", "create ",
                    "update ", "check ", "install ", "run ", "deploy "],
    "reflective":  ["i wonder", "i learned", "i noticed", "i realized", "what if",
                    "maybe we", "thinking about", "perhaps", "i feel like", "it seems"],
    "questioning": ["how do", "why does", "what is", "when should", "which one",
                    "how should", "can we", "is it possible", "what would"],
    "clarifying":  ["i mean", "to clarify", "in other words", "actually,",
                    "let me rephrase", "what i meant", "more specifically"],
    "emotional":   ["frustrated", "confused", "excited", "worried", "not sure",
                    "uncertain", "unclear", "struggling", "overwhelmed", "stuck"],
    "confirmatory":["yes,", "exactly", "that's right", "good point",
                    "agreed", "perfect", "sounds good", "makes sense"],
}

# Shifts between these pairs are considered noteworthy.
_NOTEWORTHY_TRANSITIONS = {
    ("task", "reflective"),
    ("reflective", "task"),
    ("questioning", "reflective"),
    ("task", "emotional"),
    ("emotional", "task"),
    ("confirmatory", "questioning"),
}


_INTENT_PRIORITY = ["reflective", "emotional", "questioning", "clarifying",
                     "confirmatory", "task", "general"]


def _classify_intent(text: str) -> str:
    """Return the dominant intent for a single turn using keyword signals.

    Multi-word phrases score proportionally to word count so they outweigh
    single-word matches. Ties break by _INTENT_PRIORITY ordering.
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {intent: 0.0 for intent in _INTENT_SIGNALS}
    for intent, signals in _INTENT_SIGNALS.items():
        for signal in signals:
            if signal in text_lower:
                scores[intent] += len(signal.split())
    # bare "?" is a strong questioning signal
    if "?" in text:
        scores["questioning"] += 2
    top_score = max(scores.values())
    if top_score == 0:
        return "general"
    # among tied intents, prefer higher-priority one
    for intent in _INTENT_PRIORITY:
        if scores.get(intent, 0) == top_score:
            return intent
    return "general"


class ConversationTracker:
    """Track intent and tone across turns in a single conversation session.

    Usage::

        tracker = ConversationTracker(config)
        tracker.add_turn("user", "Can you fix the ingest bug?")
        tracker.add_turn("aeon", "Sure, here's the fix.")
        tracker.add_turn("user", "Actually, I wonder if we need to rethink the whole flow.")
        result = tracker.close()   # stores arc as episodic memory and returns it

    Args:
        config:     Optional Config. Defaults to Config().
        session_id: Optional stable ID for this conversation session.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        session_id: Optional[str] = None,
    ) -> None:
        from .memory_store import _generate_id
        self.config = config or Config()
        self.session_id = session_id or _generate_id()
        self._turns: List[Dict] = []          # {speaker, text, intent}
        self._shifts: List[Dict] = []         # {from_intent, to_intent, at_turn, speaker, snippet}
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, speaker: str, text: str) -> Dict:
        """Record one turn and return its classification.

        Args:
            speaker: Who spoke — e.g. "user" or "aeon".
            text:    The message content.

        Returns:
            Dict with keys: turn_index, speaker, intent, shift_detected.
        """
        if self._closed:
            raise RuntimeError("ConversationTracker is closed; create a new instance.")

        intent = _classify_intent(text)
        turn_index = len(self._turns)

        self._turns.append({"speaker": speaker, "text": text, "intent": intent})

        shift = self._detect_shift(turn_index, intent, speaker, text)
        return {"turn_index": turn_index, "speaker": speaker, "intent": intent,
                "shift_detected": shift is not None}

    def get_arc(self) -> Dict:
        """Return a structured summary of the conversation so far (non-destructive)."""
        dominant = self._dominant_intent()
        arc = {
            "session_id":      self.session_id,
            "turn_count":      len(self._turns),
            "shifts":          list(self._shifts),
            "dominant_intent": dominant,
        }
        arc["summary"] = _build_rule_based_arc_text(arc)
        return arc

    def close(self, store: bool = True) -> Dict:
        """Finalise the session and optionally ingest the arc as an episodic memory.

        Args:
            store: If True (default), calls ingest() to persist the arc.
                   Set False to get the arc dict without writing anything.

        Returns:
            Dict with keys: arc (the arc dict) and ingest_result (or None).
        """
        if self._closed:
            raise RuntimeError("ConversationTracker already closed.")
        self._closed = True

        arc = self.get_arc()

        if not store:
            return {"arc": arc, "ingest_result": None}

        min_turns = int(getattr(self.config, "conversation_arc_min_turns", 3))
        min_shifts = int(getattr(self.config, "conversation_arc_min_shifts", 0))

        if arc["turn_count"] < min_turns or len(arc["shifts"]) < min_shifts:
            return {"arc": arc, "ingest_result": None}

        from .ingest import ingest
        text = _arc_to_ingest_text(arc, self.config)
        result = ingest(text, source="conversation", config=self.config)
        return {"arc": arc, "ingest_result": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_shift(
        self, turn_index: int, intent: str, speaker: str, text: str
    ) -> Optional[Dict]:
        if turn_index < 2:
            return None

        # Dominant intent of the two turns before this one.
        prior = self._turns[-3:-1]   # last 2 before current (already appended)
        prior_intents = [t["intent"] for t in prior]
        prev_dominant = max(set(prior_intents), key=prior_intents.count)

        if intent == prev_dominant:
            return None

        shift: Dict = {
            "from_intent": prev_dominant,
            "to_intent":   intent,
            "at_turn":     turn_index,
            "speaker":     speaker,
            "snippet":     text[:80].strip(),
            "noteworthy":  (prev_dominant, intent) in _NOTEWORTHY_TRANSITIONS,
        }
        self._shifts.append(shift)
        return shift

    def _dominant_intent(self) -> str:
        if not self._turns:
            return "general"
        counts: Dict[str, int] = {}
        for t in self._turns:
            counts[t["intent"]] = counts.get(t["intent"], 0) + 1
        return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def _arc_to_ingest_text(arc: Dict, config: Optional[Config] = None) -> str:
    """Build ingest-ready text from a completed arc dict.

    Uses LLM narrative when available; always falls back to rule-based.
    """
    rule_text = _build_rule_based_arc_text(arc)

    if config is not None and config.llm_enabled:
        from .llm import build_conversation_arc_prompt, generate_text
        prompt = build_conversation_arc_prompt(arc)
        llm_text = generate_text(prompt, config)
        if llm_text and llm_text.strip():
            return llm_text.strip()

    return rule_text


def _build_rule_based_arc_text(arc: Dict) -> str:
    lines = [
        f"Conversation arc (session {arc['session_id']}): "
        f"{arc['turn_count']} turns, {len(arc['shifts'])} intent shift(s) detected.",
        f"Dominant intent: {arc['dominant_intent']}.",
    ]

    if arc["shifts"]:
        shift_parts = []
        for s in arc["shifts"]:
            note = " [noteworthy]" if s.get("noteworthy") else ""
            shift_parts.append(
                f"{s['from_intent']} → {s['to_intent']} at turn {s['at_turn']}{note}"
            )
        lines.append("I noticed intent shifts: " + "; ".join(shift_parts) + ".")
        # Surface the trigger text for the first noteworthy shift (feeds importance scorer)
        for s in arc["shifts"]:
            if s.get("noteworthy") and s.get("snippet"):
                lines.append(f"Key moment: \"{s['snippet']}\"")
                break

    lines.append(
        "This arc captures how the user's needs evolved during the session "
        "and is stored for reflection."
    )
    return "\n".join(lines)


# Re-exported so callers can classify a single string without instantiating a tracker.
classify_intent = _classify_intent
