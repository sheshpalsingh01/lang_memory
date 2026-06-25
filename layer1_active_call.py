"""
layer1_active_call.py
─────────────────────
LAYER 1 — ACTIVE CALL MEMORY (lives only while the phone is ringing).

The agent's working memory for the call happening RIGHT NOW. In-RAM, thrown away
when the call ends (layer 3 writes the lasting parts).

It holds, turn-by-turn:
  • conversation buffer — every Human/AI turn (kept in full for the end-of-call summary)
  • intent / entities / slot values — the STRUCTURED key points of the call

Two things matter for a live voice agent:
  • how much history to put in the LLM prompt → recent_messages() returns a
    TOKEN-BUDGETED window (recent turns win) instead of the whole growing call.
  • how to know the key points without an LLM call → key_points() returns the
    intent + filled slots + entities, i.e. the salient state, at zero latency.
"""

import tiktoken
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

_ENC = tiktoken.get_encoding("cl100k_base")   # shared, just for budgeting


class ActiveCallMemory:
    def __init__(self, caller_id: str, required_slots: list[str] | None = None,
                 max_context_tokens: int = 3000):
        self.caller_id = caller_id
        self.buffer = InMemoryChatMessageHistory()   # full call history (RAM)
        self.max_context_tokens = max_context_tokens
        self.intent: str | None = None
        self.entities: dict = {}
        self.slots: dict = {name: None for name in (required_slots or [])}

    # ── conversation buffer ──────────────────────────────────────────────────
    # Skip empty / whitespace turns — STT often emits blank frames on silence, and
    # blank turns would only pollute the transcript and the end-of-call summary.
    def add_caller(self, text: str):
        if text and text.strip():
            self.buffer.add_message(HumanMessage(content=text.strip()))

    def add_agent(self, text: str):
        if text and text.strip():
            self.buffer.add_message(AIMessage(content=text.strip()))

    def messages(self):
        """The full call, oldest → newest."""
        return self.buffer.messages

    def recent_messages(self, max_tokens: int | None = None):
        """
        The most recent turns that fit the token budget — this is what you put in
        the LLM prompt. Older turns are dropped from the PROMPT only; they stay in
        the buffer so the end-of-call summary still sees the whole conversation.
        """
        budget = max_tokens if max_tokens is not None else self.max_context_tokens
        out, used = [], 0
        for m in reversed(self.buffer.messages):          # newest first
            cost = len(_ENC.encode(m.content or ""))
            if used + cost > budget and out:              # keep at least one turn
                break
            out.append(m); used += cost
        return list(reversed(out))                        # back to chronological

    def transcript(self) -> str:
        """Flat 'Caller: .. / Agent: ..' text of the WHOLE call — for summarising."""
        roletags = {"human": "Caller", "ai": "Agent"}
        return "\n".join(
            f"{roletags.get(m.type, m.type)}: {m.content}" for m in self.buffer.messages
        )

    # ── intent / entities / slots ────────────────────────────────────────────
    def set_intent(self, intent: str):
        self.intent = intent

    def update_entities(self, **kv):
        self.entities.update(kv)

    def set_slot(self, name: str, value):
        self.slots[name] = value

    def missing_slots(self) -> list[str]:
        return [name for name, val in self.slots.items() if val in (None, "")]

    def is_complete(self) -> bool:
        return not self.missing_slots()

    # ── key points (the salient state, no LLM, no latency) ───────────────────
    def key_points(self) -> dict:
        """Structured summary of where the call is: intent, filled slots, entities."""
        return {
            "intent": self.intent,
            "slots": {k: v for k, v in self.slots.items() if v not in (None, "")},
            "missing_slots": self.missing_slots(),
            "entities": dict(self.entities),
        }

    def state_summary(self) -> str:
        """One-line key-point summary to drop into the LLM system prompt."""
        kp = self.key_points()
        parts = []
        if kp["intent"]:
            parts.append(f"intent={kp['intent']}")
        if kp["slots"]:
            parts.append("known=" + ", ".join(f"{k}:{v}" for k, v in kp["slots"].items()))
        if kp["missing_slots"]:
            parts.append("still need=" + ", ".join(kp["missing_slots"]))
        if kp["entities"]:
            parts.append("entities=" + ", ".join(f"{k}:{v}" for k, v in kp["entities"].items()))
        return " | ".join(parts) if parts else "no key points yet"
