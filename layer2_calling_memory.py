"""
layer2_calling_memory.py
────────────────────────
LAYER 2 — CALLING MEMORY (everything we know about ONE caller, across calls).

Persistent and per-caller. When the same person calls again, this is what makes
the agent say "last time you asked about the 2 PM slot — still interested?".

Parts:
  • profile store      — stable facts: name, phone, language, account id …
  • episodic memory    — one searchable entry per past call, retrieved with the
                         Generative-Agents memory stream: a recalled memory wins
                         on RELEVANCE × RECENCY × IMPORTANCE, not relevance alone,
                         so a fresh, high-stakes call beats a stale near-duplicate
  • preference learning — soft learned likes/dislikes, either set explicitly or
                         distilled from past calls by reflect()
"""

import json
import math
import os
import re
import time
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from .config import settings, get_llm
from .vectorstore import make_store

# How much a call's outcome makes it "worth remembering" (Generative-Agents importance).
_IMPORTANCE = {"booked": 1.0, "callback": 0.8, "not_interested": 0.6,"failed": 0.5, "info_only": 0.3}
_RECENCY_DECAY = 0.02   # per day; exp(-decay*age) → ~35 days to half-weight

# Memory-stream recall weights — tune to bias what gets recalled. Equal by default;
# e.g. raise _W_RECENCY for a domain where the latest call matters most.
_W_RELEVANCE, _W_RECENCY, _W_IMPORTANCE = 1.0, 1.0, 1.0


def _safe(caller_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(caller_id).strip()) or "anonymous"


class CallingMemory:
    def __init__(self, caller_id: str, campaign_id: str = "default"):
        self.caller_id = _safe(caller_id)
        self.campaign_id = _safe(campaign_id)
        # SEMI-ISOLATION across campaigns:
        #  • profile/preferences are keyed by caller_id ONLY → SHARED across campaigns
        #    (a person's name/language is the same everywhere).
        #  • episodic memory (conversation history) is keyed by campaign+caller → PRIVATE
        #    per campaign (campaign A never sees campaign B's calls).
        self._profile_path = Path(settings.DATA_DIR) / "profiles" / f"{self.caller_id}.json"
        self.episodes = make_store(f"episodes_{self.campaign_id}__{self.caller_id}")
        self._data = self._load()  # {"profile": {...}, "preferences": {...}}

    # ── persistence for profile + preferences (Postgres row or local file) ────
    def _load(self) -> dict:
        if settings.DATABASE_URL:
            from .pg_store import pg_load_profile
            return pg_load_profile(self.caller_id)
        if self._profile_path.exists():
            try:
                return json.loads(self._profile_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"profile": {}, "preferences": {}}

    def _save(self):
        if settings.DATABASE_URL:
            from .pg_store import pg_save_profile
            pg_save_profile(self.caller_id, self._data)
            return
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write can't corrupt the caller's profile.
        tmp = self._profile_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self._profile_path)

    # ── profile store ────────────────────────────────────────────────────────
    def get_profile(self) -> dict:
        return dict(self._data["profile"])

    def update_profile(self, **fields):
        self._data["profile"].update(fields)
        self._save()

    # ── preference learning ──────────────────────────────────────────────────
    def learn_preference(self, key: str, value):
        self._data["preferences"][key] = value
        self._save()

    def get_preferences(self) -> dict:
        return dict(self._data["preferences"])

    # ── case state (a persistent structured record for a task, e.g. a KYC case) ─
    # Stored inside the profile under `name`. set_case MERGES the given fields into
    # the existing record (partial update), so an agent can fill it in over several
    # calls without overwriting what it already knows.
    def set_case(self, name: str = "case", **fields) -> dict:
        case = dict(self._data["profile"].get(name, {}))
        case.update(fields)
        self.update_profile(**{name: case})
        return case

    def get_case(self, name: str = "case") -> dict:
        return dict(self._data["profile"].get(name, {}))

    # ── right to erasure ─────────────────────────────────────────────────────
    def forget(self):
        """Erase this caller's stored memory: the shared profile + THIS campaign's
        episodes + this caller's global contributions (matched by metadata.caller_id)."""
        if settings.DATABASE_URL:
            from .pg_store import pg_forget
            pg_forget(self.caller_id, self.episodes.name)
        else:
            import shutil
            self._profile_path.unlink(missing_ok=True)
            shutil.rmtree(self.episodes.dir, ignore_errors=True)
        self._data = {"profile": {}, "preferences": {}}
        self.episodes = make_store(f"episodes_{self.campaign_id}__{self.caller_id}")  # fresh, empty

    # ── episodic memory ──────────────────────────────────────────────────────
    def add_episode(self, summary: str, metadata: dict | None = None):
        """Store a finished call's summary, tagged with time + importance."""
        meta = dict(metadata or {})
        meta.setdefault("ts", time.time())
        meta.setdefault("importance", _IMPORTANCE.get(meta.get("outcome"), 0.5))
        self.episodes.add(summary, meta)

    def recall_episodes(self, query: str, k: int = 3, now: float | None = None):
        """
        Memory-stream recall: over-fetch by relevance, then re-score each candidate
        as relevance + recency + importance (equal weights, normalised relevance).
        """
        cand = self.episodes.recall(query, k=max(k * 4, 8))
        if not cand:
            return []
        
        now = now if now is not None else time.time()
        rels = [d.metadata.get("_relevance", 0.0) for d in cand]
        lo, hi = min(rels), max(rels)
        norm = (lambda x: 0.0) if hi == lo else (lambda x: (x - lo) / (hi - lo))

        scored = []
        for d in cand:
            relevance = norm(d.metadata.get("_relevance", 0.0))
            age_days = max(0.0, (now - d.metadata.get("ts", now)) / 86400)
            recency = math.exp(-_RECENCY_DECAY * age_days)
            importance = float(d.metadata.get("importance", 0.5))
            score = _W_RELEVANCE * relevance + _W_RECENCY * recency + _W_IMPORTANCE * importance
            scored.append((score, d))
            
        scored.sort(key=lambda p: p[0], reverse=True)
        return [d for _, d in scored[:k]]

    # ── reflection (Generative-Agents) ───────────────────────────────────────
    def reflect(self, max_prefs: int = 3) -> dict:
        """Distil durable preferences from past calls and store them. Uses the LLM."""
        texts = [d["text"] for d in self.episodes.docs]
        if not texts:
            return self.get_preferences()
        msgs = [
            SystemMessage(content=(
                f"From these past call summaries, list up to {max_prefs} DURABLE caller "
                "preferences as 'key: value' lines (e.g. 'contact_channel: sms'). "
                "Only stable preferences, not one-off events. No other text."
            )),
            HumanMessage(content="\n".join(texts)),
        ]
        for line in get_llm().invoke(msgs).content.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = re.sub(r"[^a-z0-9_]", "", key.strip().lower().replace(" ", "_"))
                val = val.strip().strip("-* ").strip()
                if key and val:
                    self.learn_preference(key, val)
        return self.get_preferences()
