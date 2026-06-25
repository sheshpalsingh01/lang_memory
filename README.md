# lang_memory — 3-layer memory for an AI voice calling agent

Built on LangChain (`ChatGroq` for reasoning, local `bge` embeddings + FAISS for
vectors). Retrieval is **state-of-the-art, not single-vector**:

- **Hybrid + rerank** — every store fuses dense (FAISS) **+** sparse (BM25) with
  RRF, then a **cross-encoder reranks**. Fixes dense-only's blind spot on exact /
  numeric terms ("2 year" vs "3 year").
- **Memory stream** (Generative Agents) — episodic recall scores
  *relevance × recency × importance*, so a fresh, high-stakes call beats a stale
  near-duplicate.
- **Reflection** — `CallingMemory.reflect()` distils durable preferences from past
  calls automatically.

Three layers, mapped exactly to the call lifecycle:

| Layer | Class | Holds | Lifetime |
|-------|-------|-------|----------|
| 1 · Active call | `ActiveCallMemory` | buffer (token-budgeted prompt window), intent, entities, slots, `key_points()` | the live call only (in-RAM) |
| 2 · Calling memory | `CallingMemory` | profile store, episodic memory, preference learning | per caller, on disk |
| 3 · Global knowledge | `GlobalKnowledge` | call summary, outcome labeller, memory writer, failure patterns, winning scripts, RAG knowledge base | shared across all callers |

`VoiceAgentMemory` wires all three across a call's `start_call → … → end_call`.

## Install
```bash
pip install -r lang_memory/requirements.txt
# set GROQ_API_KEY in .env  (only layer 3 needs it; layers 1 & 2 run offline)
```

## Use
```python
from lang_memory import VoiceAgentMemory

agent = VoiceAgentMemory(required_slots=["date", "time"])

call = agent.start_call("+91-98765-43210")          # layer 1 + caller's layer 2
print(agent.context_for_opening(call))              # profile + preferences to personalise

call.add_caller("Book a product demo for Monday afternoon")
call.set_intent("book_demo")
call.set_slot("date", "2026-07-01")
call.set_slot("time", "15:00")
call.add_agent("Booked for Monday 3 PM, confirmation by SMS.")
print(call.is_complete())                           # True once all slots filled

agent.end_call(call)   # layer 3: summarise + label outcome + write to memory
```

On the next call from the same number, `call.calling.recall_episodes("demo?")`
returns the earlier summary, and `agent.knowledge.recall_winning_scripts(...)`
serves phrasings that previously led to a booking.

## What writes what
`end_call()` runs the **memory writer**: it summarises the transcript, labels the
outcome (`booked` / `callback` / `not_interested` / `info_only` / `failed`),
stores the summary in the caller's **episodic memory** (layer 2), and routes the
call to **winning scripts** (good outcomes) or **failure patterns** (bad ones)
in layer 3. The **RAG knowledge base** is seeded separately via
`agent.knowledge.add_knowledge(...)` and queried with `kb_search(...)`.

## Storage backends
File by default (zero setup). Set `DATABASE_URL` to switch the whole package to
**Postgres + pgvector** — same API, no code changes:

```bash
pip install "psycopg[binary]" pgvector          # + a Postgres with the pgvector extension
export DATABASE_URL=postgresql://user:pw@host:5432/langmem
```
Profiles/preferences/case → a JSONB row per caller; episodic + global stores → one
`memory_vectors` table doing hybrid retrieval **in** Postgres (pgvector dense + `tsvector`
full-text + RRF + the same cross-encoder rerank). This removes the file backend's
concurrency limit (row-level instead of a shared JSON file). Tables auto-create on first use.

## PII & compliance
Data-minimisation + encryption + erasure, so real document numbers are never sitting in
plaintext:
- **Redaction** — Layer 3 scrubs PAN/Aadhaar/passport/email from summaries and stored
  transcripts (`crypto.redact_pii`), so the vector stores hold no raw PII.
- **Field encryption** — canonical PII values (e.g. `pan_no`) are AES-encrypted at rest via
  `crypto.encrypt/decrypt` (Fernet; key from `LANG_MEMORY_ENC_KEY`). The KYC demo decrypts them
  only for an identity-verified caller.
- **Right to erasure** — `memory.forget(caller_id)` deletes a caller's profile, episodes, and
  global contributions (both backends).
- **Retention** — `memory.purge_older_than(days)` (or `RETENTION_DAYS`) drops stale records.

## Tests
A `pytest` suite lives in `tests/`. The offline tests use local embeddings (no API) and are
deterministic; the LLM tests are opt-in.

```bash
pytest lang_memory/tests                    # 31 fast offline tests (layers, store, isolation, guards)
RUN_LLM_TESTS=1 pytest lang_memory/tests    # + the 4 Groq-backed tests (summary / label / end-to-end)
```
Each test runs in its own temp dir (no shared state). Covers: Layer-1 windowing/key-points/empty
guards, HybridStore add/recall/persistence/rerank, Layer-2 profile/preferences/case-merge/episodic/
isolation/atomic-writes, Layer-3 KB + the self-learning `guidance` loop, and the orchestrator.

## Benchmark
`benchmark.ipynb` compares each layer/technique against the naive thing people
actually build (flat store, raw-history dump, keyword rules, no-RAG, single-vector,
relevance-only). Every baseline really runs — no hardcoded zeros. Headline gaps:
per-caller recall 38→100%, context tokens −62%, outcome accuracy 33→100%, KB
grounding 0→100%, numeric retrieval 75→100%, fresh-call recall 0→100%.

## Pipecat integration
`pipecat_integration.py` (`pip install pipecat-ai`) exposes `LangMemoryProcessor`, a
`FrameProcessor` you drop into a live voice pipeline (after STT, before the LLM). It
records caller turns from `TranscriptionFrame`s and bot turns from `TTSTextFrame`s,
injects the caller's profile/preferences/past calls into the LLM context, can ground each
turn with live RAG, and on `End`/`CancelFrame` summarises + labels + persists the call.
Blocking work (Groq, embeddings, cross-encoder) runs via `asyncio.to_thread` so it never
stalls the audio loop. See the module docstring for pipeline placement.

**Cost model (simple, no hidden charges):** memory is added to the prompt **once** at call
start (profile + preferences + up to 2 recent call summaries) and stays fixed — **the prompt
does not grow turn by turn**. During the call nothing is added (turns are just recorded
locally for free). At call end, **one** Groq call summarises + labels the call. So the only
extra LLM cost per call is that single end-of-call call. Live mid-call RAG is **off by
default** (`inject_rag=True` to enable; even then it replaces one grounding line each turn,
never piles up).

**Latency:** `fast=True` (default) keeps any optional retrieval KB-only, no rerank (~34 ms).
Run `python3 lang_memory/latency_bench.py` to reproduce (CPU; ~10× faster on GPU).

## Files
- `config.py` — settings + cached LLM / embeddings / reranker factories
- `vectorstore.py` — `HybridStore`: dense+BM25 → RRF → cross-encoder rerank (reused by layers 2 & 3)
- `layer1_active_call.py` / `layer2_calling_memory.py` / `layer3_global_knowledge.py`
- `voice_agent_memory.py` — orchestrator (`VoiceAgentMemory`, `CallSession`)
- `pipecat_integration.py` — `LangMemoryProcessor` for live Pipecat voice pipelines
- `api.py` — FastAPI REST wrapper (integrate any agent over HTTP); see `INTEGRATION.md`
- `tests/` — the pytest suite (layers, store, isolation, PII, campaigns, API, orchestrator)

## Per-caller isolation
Each caller is keyed by a sanitised id. Profile + preferences live in
`profiles/<caller_id>.json`; episodic memory is its own index `episodes_<caller_id>`. One
caller can never see another's history — run `pytest lang_memory/tests/test_campaign_isolation.py`
to verify. (Winning scripts / failure patterns / KB are *global* on purpose — that's the
shared learning layer; per-caller learning stays private.)
