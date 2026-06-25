"""
voice_agent_memory.py
─────────────────────
Wires the three layers into the lifecycle of a single phone call.

  agent = VoiceAgentMemory(required_slots=["date", "time"])

  call = agent.start_call("+91-98765-43210")   # layer 1 + caller's layer 2 loaded
  call.add_caller("Hi, I'd like to book a demo for next Monday afternoon")
  call.set_intent("book_demo")
  call.set_slot("date", "2026-07-01")
  ...
  agent.end_call(call)                          # layer 3 summarises + persists

start_call() also surfaces what the agent already knows (profile, preferences,
past episodes) so the dialog manager can personalise the very first sentence.
"""

from .config import settings
from .layer1_active_call import ActiveCallMemory
from .layer2_calling_memory import CallingMemory
from .layer3_global_knowledge import GlobalKnowledge


class CallSession:
    """Everything bound to one live call: layer 1 state + this caller's layer 2."""

    def __init__(self, active: ActiveCallMemory, calling: CallingMemory):
        self.active = active      # layer 1 (this call)
        self.calling = calling    # layer 2 (this caller)

    # convenience pass-throughs so the dialog code talks to one object
    def add_caller(self, text):         self.active.add_caller(text)
    def add_agent(self, text):          self.active.add_agent(text)
    def set_intent(self, intent):       self.active.set_intent(intent)
    def update_entities(self, **kv):    self.active.update_entities(**kv)
    def set_slot(self, name, val):      self.active.set_slot(name, val)
    def is_complete(self):              return self.active.is_complete()
    def key_points(self):               return self.active.key_points()


class VoiceAgentMemory:
    def __init__(self, required_slots: list[str] | None = None, campaign_id: str = "default",
                 reflect_every: int = 5):
        self.required_slots = required_slots or []
        self.campaign_id = campaign_id
        # Every `reflect_every` calls, distil durable preferences from the caller's history
        # (Generative-Agents reflection). 0 disables it. It costs one extra LLM call, so we
        # do it periodically, not every call.
        self.reflect_every = reflect_every
        # Semi-isolation across campaigns: profile is SHARED per caller; episodes +
        # global learning (layer 3) are PRIVATE per campaign.
        self.knowledge = GlobalKnowledge(campaign_id)

    def start_call(self, caller_id: str) -> CallSession:
        """Open a call: fresh layer-1 buffer + this caller's persistent layer 2."""
        active = ActiveCallMemory(caller_id, self.required_slots)
        calling = CallingMemory(caller_id, self.campaign_id)
        return CallSession(active, calling)

    def context_for_opening(self, call: CallSession) -> dict:
        """What the agent already knows before it speaks (for personalisation)."""
        return {
            "profile": call.calling.get_profile(),
            "preferences": call.calling.get_preferences(),
        }

    def end_call(self, call: CallSession) -> dict:
        """
        Close the call: layer 3 summarises + labels the transcript, then the
        memory writer persists it into layer 2 (episodic) and the global stores.
        Returns {"summary", "outcome"}.
        """
        transcript = call.active.transcript()
        if not transcript.strip():
            return {"summary": "", "outcome": "failed"}
        result = self.knowledge.write_call(call.calling, transcript)
        # Periodic reflection: once this caller has accumulated another `reflect_every`
        # calls, refresh their learned preferences from the whole history.
        if self.reflect_every and len(call.calling.episodes.docs) % self.reflect_every == 0:
            call.calling.reflect()
        return result

    # ── compliance ───────────────────────────────────────────────────────────
    def forget(self, caller_id: str):
        """Right to erasure — delete everything stored for one caller."""
        self.start_call(caller_id).calling.forget()

    def purge_older_than(self, days: int) -> int:
        """Data retention — delete memory older than `days`. Returns count removed."""
        if settings.DATABASE_URL:
            from .pg_store import pg_purge_older_than
            return pg_purge_older_than(days)
        import shutil, time
        from pathlib import Path
        cutoff, removed, root = time.time() - days * 86400, 0, Path(settings.DATA_DIR)
        for p in list(root.glob("profiles/*.json")) + list(root.glob("hybrid/*")):
            try:
                if p.stat().st_mtime < cutoff:
                    shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
                    removed += 1
            except OSError:
                pass
        return removed
