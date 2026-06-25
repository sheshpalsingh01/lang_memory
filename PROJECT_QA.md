# lang_memory — 360° Project Review & Senior-Engineer Q&A

A meeting-prep doc for the 3-layer memory system powering an AI voice calling agent.
Honest about what's strong and what's still a gap. Read top-to-bottom before a design
review; jump to a section if you're asked something specific.

---

## 1. What it is (30-second pitch)

A memory system for an **AI voice calling agent**, built on LangChain (Groq LLM +
local `bge` embeddings + FAISS/pgvector). It gives the agent three kinds of memory,
mapped to the call lifecycle:

| Layer | Holds | Lifetime | Cost |
|-------|-------|----------|------|
| **1 · Active call** | live turn buffer, intent, slots, entities | the call only (RAM) | free |
| **2 · Calling memory** | per-caller profile, preferences, episodic history | per caller, durable | free reads |
| **3 · Global knowledge** | call summaries, outcome labels, winning scripts, failure patterns, RAG KB | shared across all callers | 1 LLM call at call end |

**Design north-star:** zero LLM calls per turn, one structured LLM call at call end,
hybrid retrieval (not single-vector), per-caller isolation, and a shared cross-caller
learning loop. Typed extraction (Pydantic) keeps stored facts validated.

---

## 2. Pros (what's genuinely strong)

1. **Voice-appropriate cost model.** 0 LLM calls per turn; the prompt is injected once
   at call start and **does not grow turn by turn**. Exactly 1 structured Groq call at
   call end (summary + outcome + facts together). Predictable, no hidden per-turn charges.
2. **SOTA retrieval, not single-vector.** Dense (FAISS/pgvector) + sparse (BM25/tsvector)
   → RRF fusion → cross-encoder rerank. Fixes dense-only's blind spot on exact/numeric
   terms ("2 year" vs "3 year") — critical for KYC-type data.
3. **Memory-stream episodic recall** (Generative-Agents style): relevance × recency ×
   importance, so a fresh high-stakes call beats a stale near-duplicate.
4. **Self-learning loop is closed.** End-of-call writes winning scripts / failure patterns;
   `guidance()` reads both back. The agent repeats what worked and avoids past mistakes —
   across all callers, not just one.
5. **Per-caller isolation by construction.** Each caller is a namespaced profile + its own
   episodic index. One caller can never see another's history (proven by `tests/test_campaign_isolation.py`).
6. **Typed extraction.** Pydantic `CallRecord` (summary + outcome + facts); outcome is a
   `Literal` so the model can't emit an invalid label. No fragile regex parsing.
7. **Pluggable storage.** File backend (zero setup, dev) or Postgres+pgvector (`DATABASE_URL`)
   — same API, no code changes. Hybrid retrieval runs *inside* Postgres.
8. **PII compliance built in.** Redaction (data minimization), Fernet field encryption,
   reveal-on-verify (last-4 masking), right-to-erasure (`forget`), retention (`purge`).
9. **Graceful degradation.** If Groq is down at call end, it falls back to a basic record
   instead of crashing or losing the call.
10. **Tested + CI.** pytest suite with temp-dir isolation per test; offline tests need no API;
    LLM tests opt-in. pyflakes in CI.

---

## 3. Cons / known gaps (be upfront about these)

1. **Concurrency ceiling on the file backend.** Global stores are shared JSON+FAISS files —
   concurrent writes race. File backend is dev/demo-scale only; Postgres is the production path.
2. **No connection pooling yet.** Postgres backend is connect-per-call → bounded by
   `max_connections (~100)`. A ~6-line `psycopg_pool` change fixes it; not done.
3. **In-RAM session dict isn't shared across workers.** Multi-worker uvicorn would 404 a call
   started on another worker. Needs sticky routing or Redis-backed sessions for horizontal scale.
4. **Hardcoded Groq key fallback in config.** Convenient for dev, must be env-only + rotated
   before production. Flagged repeatedly.
5. **No observability.** No metrics/tracing/structured logs. You can't yet see p99 latency,
   LLM error rate, or recall quality in prod.
6. **Retrieval quality not continuously evaluated.** We benchmarked once; there's no online
   eval / regression guard on recall as data grows.
7. **Single embedding + reranker model, CPU-bound.** Fine at current scale; rerank is the
   latency cost. GPU or a hosted reranker needed at high throughput.
8. **No auth/rate-limiting on the REST API.** It's an internal service as written.
9. **Outcome label set is fixed (5 values).** Domain changes need code edits, not config.
10. **`langchain-community` deprecation warning** (FAISS import) — migration debt.

---

## 4. Senior-engineer questions, by category (with honest answers)

### A. Architecture & design

**Q: Why three layers instead of one store?**
Because they have different lifetimes, costs, and access patterns. Layer 1 is ephemeral and
free (RAM); Layer 2 is per-caller durable; Layer 3 is shared learning. Collapsing them either
re-introduces per-turn LLM cost (mem0-style) or loses the cross-caller learning layer.

**Q: Why not just use mem0 / LangMem?**
They're text-chat memory patterns: extract + dedup on **every turn**, semantic-only recall.
That re-adds per-turn LLM latency (we measured mem0 hitting Groq's TPM limit) and mis-handles
exact/numeric facts. We took their one good idea — typed extraction — and grafted it onto a
voice-optimized design (extraction at call end, hybrid retrieval).

**Q: Where does extraction happen and why there?**
At call end, in one structured call. During the call, turns are just appended to RAM. This
keeps the live audio loop free of LLM latency — the single most important voice constraint.

**Q: What's the upsert/dedup strategy?**
Deterministic key match (`caller_id + field`), not an LLM "is this already known?" check.
Scalar facts overwrite by key (last-write-wins); blanks never wipe known values. Cheap,
deterministic, debuggable.

### B. Scaling & concurrency

**Q: How many concurrent users?**
- Open calls held in RAM: thousands (cheap dict).
- In-flight ops doing work: ~40 (Starlette threadpool; heavy ops release the GIL).
- Real ceiling = backend: file ≈ a handful safely; Postgres (no pool) ≈ dozens; Postgres +
  pool + Redis sessions + N workers ≈ hundreds–thousands.

**Q: What breaks first under load?**
Either shared-file write contention (file backend) or Postgres connection exhaustion (no pool).
Then CPU on the cross-encoder rerank.

**Q: How do you scale horizontally?**
Order: Postgres backend → connection pool → externalize the session dict to Redis →
multiple uvicorn workers behind a load balancer. The session dict is the blocker today.

### C. Latency

**Q: What's the hot-path latency?**
~34 ms with `fast=True` (KB-only, no rerank). Cross-encoder rerank pushes optional retrieval
to ~90 ms. Live RAG is off by default so turns add nothing. The only ~0.5–1s cost is the
single end-of-call LLM call, which happens after the conversation.

**Q: Does the cross-encoder block the audio loop?**
In the Pipecat integration, blocking work (Groq, embeddings, rerank) runs via
`asyncio.to_thread`, so it never stalls the audio event loop.

### D. Cost

**Q: If my system prompt is 2k tokens, does it grow during the call?**
No. Memory is injected once at call start (profile + prefs + ≤2 past summaries) and stays
fixed. Turns are recorded locally for free. One structured LLM call at call end. That's the
entire extra LLM spend per call.

**Q: What's the cost driver at scale?**
End-of-call calls × call volume, plus embedding compute. Both are linear and predictable.

### E. Data, storage & retrieval

**Q: Why hybrid + rerank instead of pure vector search?**
Semantic-only misses exact/numeric matches and produces dupes/dropped writes on the
"already known?" check. Hybrid (dense+sparse → RRF → rerank) catches both fuzzy and exact.

**Q: How is episodic recall ranked?**
Memory-stream: over-fetch by relevance, re-score as relevance + recency + importance
(importance keyed off outcome). Recent, important calls win over stale near-duplicates.

**Q: File vs Postgres — what actually changes?**
Same API. Postgres moves hybrid retrieval into SQL (pgvector + tsvector + RRF + same rerank),
removes the shared-file race (row-level), and scales. Tables auto-create on first use.

**Q: How do you isolate multiple campaigns? (multi-tenancy)**
Semi-isolation by `campaign_id`: the caller **profile is SHARED** across campaigns (a person's
name/language is the same everywhere), but **episodic history and the global learning layer
(winning scripts / failure patterns / KB) are PRIVATE per campaign** — campaign A never sees
campaign B's calls or learned scripts. Pass `VoiceAgentMemory(campaign_id=...)` (or `campaign_id`
in the `/calls` request). Profiles key on `caller_id`; episodes/learning key on `campaign:caller`.

### F. Security & PII (likely the hardest questions)

**Q: Where does raw PII live?**
Canonical doc numbers (PAN/Aadhaar/passport) are **Fernet-encrypted at rest**. Free-text
summaries and stored transcripts are **redacted** (PAN/Aadhaar/passport/email → tags), so the
vector stores hold no raw PII.

**Q: Can a stranger on a known number extract stored details?**
No. Fields are masked until identity is verified. Unverified callers see only the **last 4**
(bank-style confirmation); full reveal requires a security question + `verify_identity`.

**Q: Right to erasure / retention?**
`forget(caller_id)` deletes profile + episodes + that caller's global contributions (both
backends). `purge_older_than(days)` / `RETENTION_DAYS` drops stale records.

**Q: Biggest security gap today?**
The hardcoded Groq key fallback (must go env-only + rotate) and no API auth/rate-limiting.
Both known, both flagged.

### G. Reliability & failure modes

**Q: What if the LLM is down at call end?**
`extract_call_record` falls back to a basic record (`transcript[:300]`, outcome `failed`) —
the call is still persisted; nothing crashes or is lost.

**Q: What if STT mis-hears a document number?**
Validators (regex) reject malformed PAN/Aadhaar/passport before storage; the agent is told to
read it back and re-ask. A wrong KYC number is never silently stored.

**Q: Mid-call correction ("actually my PAN is X")?**
Last-write-wins by key, with an encrypted audit trail of what changed (KYC compliance).

### H. Testing & quality

**Q: How is it tested?**
pytest with an autouse temp-dir fixture (no shared state). Offline tests use local embeddings
(deterministic, no API); LLM tests are opt-in (`RUN_LLM_TESTS=1`). 35 offline pass; pyflakes
in CI. Covers Layer-1 windowing/guards, hybrid store add/recall/rerank, Layer-2
profile/case/episodic/isolation, Layer-3 KB + the guidance loop, PII, and the orchestrator.

**Q: How do you know the memory actually helps?**
`benchmark.ipynb` compares each technique against the naive baseline people actually build
(flat store, raw dump, keyword rules, no-RAG, single-vector, relevance-only). Every baseline
really runs. Headline gaps: per-caller recall 38→100%, context tokens −62%, outcome accuracy
33→100%, numeric retrieval 75→100%.

**Q: Gap in testing?**
No load/concurrency tests, no online retrieval-quality eval as data grows.

### I. Observability & ops

**Q: How do you debug a bad recall in prod?**
Honestly — limited today. No tracing/metrics yet. This is a known gap; first thing I'd add for
production is structured logs + latency/error metrics + a recall-quality sampler.

### J. Trade-offs you chose deliberately

1. **Extraction at call end, not per turn** — trades freshest-possible mid-call memory for
   zero per-turn latency/cost. Right call for voice.
2. **Global learning is shared, per-caller is private** — deliberate: cross-caller learning is
   the point of Layer 3; personal data stays isolated.
3. **Kept `summarize()`/`label_outcome()` as standalone utilities** even after consolidating
   the writer to one call — they back the latency benchmark; minor duplication accepted for
   measurability.
4. **File backend default** — zero-setup dev experience over out-of-box concurrency.

---

## 5. "If this went to production tomorrow, what would you do first?" (the closer)

Priority order, honestly:
1. Move the Groq key to env-only + rotate it; add API auth + rate limiting.
2. Switch to Postgres + add connection pooling.
3. Externalize the session dict to Redis so multiple workers work.
4. Add observability: structured logs, latency/error metrics, a recall-quality sampler.
5. Add load/concurrency tests and an online retrieval-eval guard.

Everything in 1–3 is small and well-understood; 4–5 is the ongoing production hygiene.

---

## 6. One-line honest self-assessment

**Strong, voice-optimized prototype (≈7–8/10).** The hard design calls (cost model, hybrid
retrieval, isolation, learning loop, PII) are right and built. The remaining gaps are
operational (pooling, sessions-in-Redis, observability, secrets) — known, scoped, and small.
