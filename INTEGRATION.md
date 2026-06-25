# Integrating `lang_memory` into any voice calling agent

This memory system is **framework-agnostic**. Whatever stack you use (Pipecat, LiveKit,
Vapi, Twilio + your own loop, a custom WebSocket agent…), integration is the **same 4
touch-points** in your call lifecycle:

| When | Call | Why |
|------|------|-----|
| ☎️ **Call starts** | `call = memory.start_call(caller_id)` | load this caller's memory; build a personalised system prompt |
| 🗣️ **Caller speaks** (final STT) | `call.add_caller(text)` | record the turn (+ optional slots/intent) |
| 🤖 **Bot speaks** | `call.add_agent(text)` | record the turn |
| 🔚 **Call ends** | `memory.end_call(call)` | summarise + label + persist everything |

That's it. Everything else (retrieval, isolation, learning) happens behind those calls.

---

## 1. Install

```bash
pip install -r lang_memory/requirements.txt
export GROQ_API_KEY=...          # only used at call-end (summary + outcome). Get one free at console.groq.com
```
Embeddings + vector search run **locally** (no API). Only the end-of-call summary uses an LLM.

---

## 2. Five-minute generic integration (works with ANY agent)

```python
from lang_memory import VoiceAgentMemory

# Create ONCE, app-wide. required_slots = the form your agent fills (optional).
memory = VoiceAgentMemory(required_slots=["date", "time"])

# (optional, once) seed the shared knowledge base the bot can quote
memory.knowledge.add_knowledge("A product demo is 30 minutes over Google Meet and is free.",
                               {"topic": "demo"})

# ── when a call connects ────────────────────────────────────────────────
call = memory.start_call(caller_phone)             # caller_phone = a STABLE id (phone / account id)
system_prompt = build_system_prompt(call)          # see §3 — feed this to your LLM

# ── on each final user transcript (from your STT) ───────────────────────
call.add_caller(user_text)
# optional: tell memory what you understood (your NLU stays in your app)
call.set_intent("book_demo")
call.set_slot("date", "2026-07-01")
if call.is_complete():                             # all required_slots filled?
    ...                                            # submit the booking

# ── on each thing the bot says ──────────────────────────────────────────
call.add_agent(bot_text)

# ── when the call ends / hangs up ───────────────────────────────────────
result = memory.end_call(call)                     # {"summary": "...", "outcome": "booked"}
```

Next time the **same `caller_phone`** calls, `start_call` already knows their profile,
preferences and past calls — that's the whole point.

---

## 3. Build the system prompt from memory

Put what the agent already knows at the top of your LLM context **once**, at call start.
It stays fixed for the whole call, so the prompt never grows turn by turn:

```python
def build_system_prompt(call, base="You are a helpful voice agent."):
    profile = call.calling.get_profile()           # {"name": "Ravi", "language": "en"}
    prefs   = call.calling.get_preferences()        # {"contact_channel": "sms"}
    past    = call.calling.recall_episodes("our previous conversations", k=2)

    lines = [base]
    if profile: lines.append("Caller: " + ", ".join(f"{k}={v}" for k, v in profile.items()))
    if prefs:   lines.append("Preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items()))
    if past:    lines.append("Past calls:\n" + "\n".join(f"- {d.page_content}" for d in past))
    return "\n".join(lines)
```

During the call you can also drop the live state into the prompt (zero latency, no LLM):

```python
call.active.state_summary()   # "intent=book_demo | known=date:2026-07-01 | still need=time"
call.active.recent_messages() # token-budgeted recent turns (won't blow up the prompt)
```

---

## 4. Pipecat (drop-in — no glue code)

If you use Pipecat, skip the manual wiring — there's a ready `FrameProcessor`:

```python
from lang_memory import VoiceAgentMemory
from lang_memory.pipecat_integration import LangMemoryProcessor

memory   = VoiceAgentMemory(required_slots=["date", "time"])
mem_proc = LangMemoryProcessor(memory, caller_id=caller_phone, context=context)
context.add_message(mem_proc.opening_memory_message())   # personalise turn 1

pipeline = Pipeline([
    transport.input(),
    stt,
    mem_proc,                       # records caller turns, persists on call end
    context_aggregator.user(),
    llm,
    tts,                            # mem_proc also records the bot's TTS text
    transport.output(),
    context_aggregator.assistant(),
])
```

It records turns from `TranscriptionFrame`/`TTSTextFrame` and persists on `End`/`CancelFrame`.
Blocking work runs off the audio loop via `asyncio.to_thread`. (Needs `pip install pipecat-ai`.)

---

## 4b. Over HTTP (any language, not just Python)

If your agent isn't Python, run the REST API and hit it over HTTP — same 4 touch-points:

```bash
pip install fastapi uvicorn
uvicorn lang_memory.api:app --port 8000
```

```
POST /calls                  {caller_id, required_slots?}     → {call_id, system_prompt, profile, preferences}
POST /calls/{id}/turn        {role: "caller"|"agent", text}   record a turn
POST /calls/{id}/profile     {profile?, preferences?}          persist caller facts (for next call)
POST /calls/{id}/state       {intent?, slots?, entities?}      push your NLU result
GET  /calls/{id}/state                                         live key points + is_complete
POST /calls/{id}/end                                           → {summary, outcome}
POST /knowledge              {text, metadata?}                 seed the RAG KB
GET  /knowledge/search?q=&k=                                   search the KB
GET  /callers/{caller_id}                                      a caller's stored profile + preferences
```

The server keeps the live call in memory and persists everything on `/end`. Full OpenAPI docs
at `http://localhost:8000/docs`.

## 5. Optional extras (use only if you need them)

**Look up a fact mid-call (RAG grounding):**
```python
docs = memory.knowledge.kb_search(user_text, k=3, rerank=False)   # rerank=False ≈ 38 ms (hot path)
grounding = "\n".join(d.page_content for d in docs)               # add to the LLM context for THIS turn
```

**Reuse phrasings that worked before:**
```python
memory.knowledge.recall_winning_scripts("how to close a hesitant caller", k=1)
```

**Auto-learn durable preferences from a caller's history (run occasionally, off-call):**
```python
call.calling.reflect()    # LLM distils "contact_channel: sms", "language: hindi", …
```

---

## 6. API cheat-sheet

```
VoiceAgentMemory(required_slots=[...])
  .start_call(caller_id) -> CallSession
  .context_for_opening(call) -> {"profile":..., "preferences":...}
  .end_call(call) -> {"summary":..., "outcome": booked|callback|not_interested|info_only|failed}
  .knowledge.add_knowledge(text, meta)         # seed RAG KB
  .knowledge.kb_search(query, k, rerank)       # search RAG KB
  .knowledge.recall_winning_scripts(query, k)

CallSession (one live call)
  .add_caller(text) / .add_agent(text)         # record turns
  .set_intent(s) / .set_slot(name, val) / .update_entities(**kv)
  .is_complete() / .key_points()
  .calling.get_profile() / .update_profile(**f)
  .calling.get_preferences() / .learn_preference(k, v)
  .calling.recall_episodes(query, k)           # this caller's past calls
  .active.recent_messages() / .state_summary() # budgeted prompt window / live state
```

---

## 7. Config & storage

- **Settings:** `lang_memory/config.py` — `GROQ_API_KEY`, `GROQ_MODEL`, `EMBEDDING_MODEL`, `DATA_DIR`.
- **Storage:** local files under `./data/lang_memory/` (set `LANG_MEMORY_DIR` to change).
  Profiles → `profiles/<caller_id>.json`; vectors → `hybrid/<store>/`. FAISS now, swap to a
  vector DB later (the single seam is `HybridStore` in `vectorstore.py`).
- **Per-caller isolation** is automatic — each `caller_id` has its own files; callers never mix.
  Use a **stable, normalised** id (e.g. E.164 phone like `+919876543210`) so the same person maps
  to the same memory every time.

## 8. Cost & latency (what to expect)

- **Per turn:** recording is free (in-RAM). Optional KB lookup ≈ **38 ms** (`rerank=False`), CPU.
- **Per call:** exactly **one** LLM call at the end (summary + outcome). Nothing per turn.
- **Prompt:** memory goes in **once** at call start and stays fixed — it does not grow per turn.
- **Cold start:** models load once (~16 s) at process boot — warm up before taking calls.
```python
from lang_memory.config import get_embeddings
get_embeddings().embed_query("warmup")   # call at startup
```
