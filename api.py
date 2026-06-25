"""
api.py
──────
HTTP wrapper around the 3-layer memory, so a voice agent in ANY language can use it
over REST. It exposes the same 4 lifecycle touch-points as the library:

  POST /calls                       → start a call (returns a personalised system prompt)
  POST /calls/{call_id}/turn        → record a caller/agent turn
  POST /calls/{call_id}/end         → summarise + label + persist the call
  GET/POST helpers for state, knowledge base, and a caller's stored profile.

The live call (layer 1) is in-RAM, so the server keeps active CallSessions in a dict
keyed by a generated call_id. Endpoints are plain `def` so FastAPI runs them in a
threadpool — the blocking embedding / LLM work never stalls the event loop.

Run:  pip install fastapi uvicorn
      uvicorn lang_memory.api:app --host 0.0.0.0 --port 8000
"""

import uuid
import time
import asyncio
import threading
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .voice_agent_memory import VoiceAgentMemory
from .prompt import build_system_prompt, scalar_fields
from .config import get_embeddings


@lru_cache(maxsize=None)
def _memory(campaign_id: str = "default") -> VoiceAgentMemory:
    """One memory per campaign (cached). Profiles are shared per caller across campaigns;
    episodes + learning are isolated per campaign — see VoiceAgentMemory."""
    return VoiceAgentMemory(campaign_id=campaign_id)


_sessions: dict = {}                           # call_id → live CallSession (layer 1 + 2)
_seen: dict = {}                               # call_id → last-activity unix time
_lock = threading.Lock()

# Abandoned-call cleanup: live calls are in RAM, so a call that never gets /end (caller
# just hangs up) would sit there forever. The reaper drops sessions idle past this TTL.
IDLE_TTL = 1800                                # 30 minutes


def _touch(call_id: str):
    _seen[call_id] = time.time()


async def _reaper():
    """Every minute, drop sessions with no activity for IDLE_TTL seconds."""
    while True:
        await asyncio.sleep(60)
        cutoff = time.time() - IDLE_TTL
        with _lock:
            stale = [cid for cid, t in _seen.items() if t < cutoff]
            for cid in stale:
                _sessions.pop(cid, None)
                _seen.pop(cid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the embedding model in the BACKGROUND so the server (and /docs) is reachable
    # immediately. It finishes loading within a few seconds; the first /calls or
    # /knowledge request just waits for it if it hasn't loaded yet.
    warm = asyncio.create_task(asyncio.to_thread(lambda: get_embeddings().embed_query("warmup")))
    reaper = asyncio.create_task(_reaper())    # abandoned-call cleanup
    yield
    warm.cancel()
    reaper.cancel()


app = FastAPI(title="lang_memory API", version="1.0", lifespan=lifespan)


# ── request/response models ──────────────────────────────────────────────────
class StartReq(BaseModel):
    caller_id: str                             # STABLE id, e.g. E.164 phone "+919876543210"
    campaign_id: str = "default"               # isolates episodes + learning per campaign
    required_slots: list[str] = []
    base_prompt: str = "You are a helpful voice agent."

class TurnReq(BaseModel):
    role: str                                  # "caller" or "agent"
    text: str

class StateReq(BaseModel):
    intent: str | None = None
    slots: dict = {}
    entities: dict = {}

class ProfileReq(BaseModel):
    profile: dict = {}                         # stable facts: {"name": "Ravi", "language": "en"}
    preferences: dict = {}                     # learned prefs: {"contact_channel": "sms"}

class KnowledgeReq(BaseModel):
    text: str
    metadata: dict = {}


# ── helpers ──────────────────────────────────────────────────────────────────
def _get(call_id: str):
    session = _sessions.get(call_id)
    if session is None:
        raise HTTPException(404, f"unknown call_id {call_id}")
    _touch(call_id)                            # any access counts as activity
    return session


# ── lifecycle ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "active_calls": len(_sessions)}

@app.post("/calls")
def start_call(req: StartReq):
    """Open a call. Returns a call_id and a ready-to-use personalised system prompt."""
    session = _memory(req.campaign_id).start_call(req.caller_id)
    if req.required_slots:
        session.active.slots = {s: None for s in req.required_slots}
    call_id = uuid.uuid4().hex
    with _lock:
        _sessions[call_id] = session
        _touch(call_id)
    return {
        "call_id": call_id,
        "caller_id": req.caller_id,
        "system_prompt": build_system_prompt(session, req.base_prompt),
        "profile": session.calling.get_profile(),
        "preferences": session.calling.get_preferences(),
    }

@app.post("/calls/{call_id}/turn")
def add_turn(call_id: str, req: TurnReq):
    """Record one turn. role = 'caller' (user STT) or 'agent' (bot speech)."""
    session = _get(call_id)
    if req.role == "caller":
        session.add_caller(req.text)
    elif req.role == "agent":
        session.add_agent(req.text)
    else:
        raise HTTPException(400, "role must be 'caller' or 'agent'")
    return {"ok": True}

@app.post("/calls/{call_id}/state")
def update_state(call_id: str, req: StateReq):
    """Optional: push your NLU result (intent / slots / entities) into the call."""
    session = _get(call_id)
    if req.intent:
        session.set_intent(req.intent)
    for name, val in req.slots.items():
        session.set_slot(name, val)
    if req.entities:
        session.update_entities(**req.entities)
    return session.key_points()

@app.post("/calls/{call_id}/profile")
def save_profile(call_id: str, req: ProfileReq):
    """Persist this caller's profile + preferences (layer 2) — used on the NEXT call."""
    session = _get(call_id)
    if req.profile:
        session.calling.update_profile(**req.profile)
    for key, val in req.preferences.items():
        session.calling.learn_preference(key, val)
    return {"profile": session.calling.get_profile(), "preferences": session.calling.get_preferences()}

@app.get("/calls/{call_id}/state")
def get_state(call_id: str):
    """The live key points (intent, filled/missing slots, entities) — zero latency."""
    session = _get(call_id)
    return {"key_points": session.key_points(), "is_complete": session.is_complete()}

@app.post("/calls/{call_id}/end")
def end_call(call_id: str):
    """Summarise + label + persist the call, then forget the live session."""
    session = _get(call_id)
    # route to the SAME campaign's memory (its layer-3 stores) the call was opened with
    result = _memory(session.calling.campaign_id).end_call(session)
    with _lock:
        _sessions.pop(call_id, None)
        _seen.pop(call_id, None)
    return result


# ── knowledge base + caller lookup ───────────────────────────────────────────
@app.post("/knowledge")
def add_knowledge(req: KnowledgeReq, campaign_id: str = "default"):
    """Seed the KB for a campaign (each campaign has its own product/FAQ knowledge)."""
    _memory(campaign_id).knowledge.add_knowledge(req.text, req.metadata)
    return {"ok": True}

@app.get("/knowledge/search")
def search_knowledge(q: str, k: int = 3, rerank: bool = True, campaign_id: str = "default"):
    docs = _memory(campaign_id).knowledge.kb_search(q, k=k, rerank=rerank)
    return {"results": [{"text": d.page_content, "metadata": {kk: vv for kk, vv in d.metadata.items()
                                                              if kk != "_relevance"}} for d in docs]}

@app.get("/callers/{caller_id}")
def get_caller(caller_id: str):
    """A caller's persistent identity + preferences (SHARED across campaigns).
    Scalar identity fields only — internal case state / encrypted PII is never exposed."""
    calling = _memory().start_call(caller_id).calling
    return {"profile": scalar_fields(calling.get_profile()), "preferences": calling.get_preferences()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
