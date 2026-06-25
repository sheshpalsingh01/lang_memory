# Integration Prompt — wire `lang_memory` into a Pipecat voice agent

> **How to use this file:** paste it into your AI coding assistant (Cursor / Claude / Copilot)
> together with your existing Pipecat bot file and say: *"Integrate lang_memory following this
> spec exactly."* It contains the **verified API** (don't invent methods), the **exact Pipecat
> wiring**, and the **pitfalls that silently break voice-agent memory** so you get it right the
> first time.

---

## 0. Role & rules for the assistant

You are integrating the local `lang_memory` package into a **Pipecat** voice agent. Obey:

1. **Only use the methods listed in §2.** Do not guess method names — they are exact.
2. **Import the processor from `lang_memory.pipecat_integration`**, NOT from `lang_memory`
   (it is intentionally not re-exported, because it imports `pipecat`).
3. **Do the verification step in §4 before trusting it.** Frame routing is the #1 source of
   silent failures — confirm which frames actually reach the processor.
4. Keep `GROQ_API_KEY` in the environment. It is used **only** at call end (summary + outcome).
5. Use a **stable `caller_id`** (E.164 phone like `+919876543210` or an account id) so the same
   person maps to the same memory every time.

---

## 1. What you are wiring (the goal)

| Lifecycle event | What must happen |
|---|---|
| Call starts | load the caller's memory, add a personalised system prompt to the LLM context |
| Caller speaks (final STT) | record the turn |
| Bot speaks | record the turn |
| Call ends | one LLM call: summarise + label + persist to the caller's memory |

The package already does all of this inside `LangMemoryProcessor`. Your job is **placement + the
opening prompt + verification.**

---

## 2. Verified API (ground truth — copy, do not deviate)

```python
# Core library
from lang_memory import VoiceAgentMemory
memory = VoiceAgentMemory(required_slots=["date", "time"])   # required_slots optional
memory.knowledge.add_knowledge(text, metadata: dict = {})    # seed RAG KB (optional)

# Pipecat processor (import from the submodule!)
from lang_memory.pipecat_integration import LangMemoryProcessor
mem = LangMemoryProcessor(
    memory,                 # the VoiceAgentMemory above
    caller_id="+91...",     # STABLE id
    context=context,        # your LLM context object (must have .add_message and .messages)
    inject_rag=False,       # default; True = also inject KB grounding each turn (advanced)
    fast=True,              # default; hot-path retrieval skips the cross-encoder (~38ms)
)
mem.opening_memory_message() -> dict   # {"role":"system","content": "...profile/prefs/past calls"}
# Handled automatically inside process_frame: StartFrame, TranscriptionFrame, TTSTextFrame,
#                                             EndFrame, CancelFrame
# After a call starts you can reach mem.session to push NLU:
mem.session.set_intent("book_demo"); mem.session.set_slot("date", "2026-07-01")
```

Do **not** call `process_frame` yourself — Pipecat does. Do **not** persist manually — the
processor does it on End/Cancel.

---

## 3. Exact Pipecat wiring

```python
from pipecat.pipeline.pipeline import Pipeline
from lang_memory import VoiceAgentMemory
from lang_memory.pipecat_integration import LangMemoryProcessor

memory = VoiceAgentMemory(required_slots=["date", "time"])
caller_id = caller_phone_number                      # from your telephony transport

# context = your LLM context (e.g. OpenAILLMContext). Create it BEFORE the processor.
mem = LangMemoryProcessor(memory, caller_id=caller_id, context=context)

# Personalise turn 1 — add the caller's memory to the context ONCE, before the run:
context.add_message(mem.opening_memory_message())

pipeline = Pipeline([
    transport.input(),
    stt,
    mem,                          # ← see §4: this position captures caller turns + lifecycle
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

---

## 4. ⚠️ The #1 pitfall — frame routing (verify, don't assume)

A processor only sees frames that **pass through its position**. With the placement above, `mem`
reliably sees `TranscriptionFrame` (caller turns) and `StartFrame`/`EndFrame`/`CancelFrame`. But
**`TTSTextFrame` (bot speech) is produced by `tts`, which is *after* `mem`** — depending on your
Pipecat version it may **not** flow back through `mem`, so **bot turns can be silently missed**,
making the end-of-call summary one-sided.

**Verify before trusting** — temporarily log every frame the processor sees:

```python
# add at the top of LangMemoryProcessor.process_frame, or subclass for the test:
print("FRAME:", type(frame).__name__)
```

Run one real call and check the log contains BOTH `TranscriptionFrame` AND `TTSTextFrame`.

**If `TTSTextFrame` is missing, pick ONE fix:**
- **A (simplest):** move `mem` to **just before `transport.output()`** (after `tts`). A frame
  travelling downstream passes through every later processor, so this position sees caller
  transcripts *and* bot TTS text. (Trade-off: live `inject_rag` won't ground the current turn from
  here — fine, since `inject_rag=False` by default.)
- **B (keep live RAG):** keep `mem` before the aggregator for injection, and record bot turns from
  the assistant side — call `mem.session.add_agent(text)` wherever you already have the bot's final
  text (e.g. an `LLMFullResponseEndFrame` handler or `context_aggregator.assistant()` output).

Default recommendation for **minimum error**: **inject_rag=False + place `mem` right before
`transport.output()`** → one processor captures both sides cleanly.

---

## 5. Other pitfalls → rules

1. **Frame names are version-sensitive.** Confirm against your install:
   `python -c "import pipecat.frames.frames as f; print([n for n in dir(f) if 'Frame' in n])"`.
   If `TranscriptionFrame`/`TTSTextFrame` differ, update the imports in `pipecat_integration.py`.
2. **Context object contract:** the `context` you pass must have `.add_message(dict)` and a
   `.messages` list (true for `OpenAILLMContext`). If yours differs, opening injection / rolling
   RAG won't apply — adapt `opening_memory_message()` usage accordingly.
3. **Don't let memory grow the prompt.** Add the opening message **once** (already handled at
   `StartFrame`). Do not re-add memory each turn.
4. **Blocking work is already off-loop** (`asyncio.to_thread`) — don't wrap calls again.
5. **Warm up at startup** so call 1 isn't a ~16 s cold start:
   `from lang_memory.config import get_embeddings; get_embeddings().embed_query("warmup")`.
6. **GROQ_API_KEY missing** → `end_call` raises. Set it; it's only needed at call end.
7. **Per-caller isolation is automatic** — never share one `caller_id` across people.

---

## 6. Acceptance checklist (the integration is correct when…)

- [ ] `import`s resolve; processor comes from `lang_memory.pipecat_integration`.
- [ ] Frame log shows BOTH `TranscriptionFrame` and `TTSTextFrame` reaching the processor.
- [ ] Turn 1 system prompt contains the caller's profile/preferences/past calls (returning caller).
- [ ] After a call, `memory.start_call(caller_id).calling.recall_episodes("...")` returns its summary.
- [ ] A second call from the same `caller_id` is personalised; a different `caller_id` sees nothing of the first.
- [ ] Prompt size does not grow per turn.
- [ ] No `GROQ_API_KEY` errors; end-of-call returns `{"summary","outcome"}`.

---

## 7. Minimal smoke test (no telephony needed)

```python
# Proves the memory half works without Pipecat services.
from lang_memory import VoiceAgentMemory
m = VoiceAgentMemory(required_slots=["date"])
c = m.start_call("+919876543210")
c.calling.update_profile(name="Ravi"); c.calling.learn_preference("contact_channel","sms")
c.add_caller("book a demo monday"); c.set_slot("date","2026-07-01")
c.add_agent("Booked Monday 3 PM, Ravi.")
print(m.end_call(c))                                  # {"summary":..., "outcome":"booked"}
print(m.start_call("+919876543210").calling.get_profile())   # {"name":"Ravi"} → persisted
```
If this prints a summary and the persisted profile, the memory layer is healthy and any
remaining issue is purely Pipecat frame wiring (§4).
```
