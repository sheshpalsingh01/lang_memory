# lang_memory — End-to-End Workflow

How a call flows through all three layers, start to finish. Diagrams first, then the
real KYC demo output mapped to each step.

---

## 1. The big picture (one call)

```
        ┌──────────────────────────── ONE PHONE CALL ───────────────────────────┐
        │                                                                        │
  start_call(caller_id, campaign)                                          end_call(call)
        │                                                                        │
        ▼                                                                        ▼
 ┌───────────────┐      every turn (NO LLM)        ┌───────────────┐    1 STRUCTURED LLM CALL
 │  FETCH from    │   ───────────────────────►     │  RECORD turns  │   ───────────────────────►
 │  memory ONCE   │                                 │  in Layer 1    │
 └───────┬────────┘                                 │  (RAM buffer)  │
         │                                          └───────┬────────┘
         │ profile + case + 2 summaries                     │ live turns
         ▼                                                  ▼
 ┌──────────────────────────┐                    ┌────────────────────────────┐
 │ SYSTEM PROMPT (built once)│                    │ extract_call_record():     │
 │  • profile (shared)       │                    │   summary + outcome + facts│
 │  • resumed case (missing) │                    └─────────────┬──────────────┘
 │  • ≤2 past summaries      │                                  │ writes
 │  FIXED — does not grow    │                                  ▼
 └──────────────────────────┘                    ┌────────────────────────────┐
                                                  │ Layer 2: profile facts +   │
                                                  │          episode summary   │
                                                  │ Layer 3: winning / failure │
                                                  │          (per campaign)    │
                                                  └────────────────────────────┘
```

**Key:** memory is fetched **once** at start (fixed prompt), turns cost **0 LLM calls**,
and exactly **1 structured LLM call** persists everything at the end.

---

## 2. Where each layer lives

```
  ┌─────────────────────── APP WORKER (process) ───────────────────────┐
  │                                                                     │
  │  LAYER 1  Active call   →  RAM (the live turn buffer)  [ephemeral]  │
  │  LAYER 2  Calling mem   →  logic here, DATA in ▼                     │
  │  LAYER 3  Global know.  →  logic + models here, DATA in ▼           │
  │                                                                     │
  └──────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │  POSTGRES (or local files in dev) │
                │   • memory_profiles  (Layer 2)    │   profile = SHARED per caller
                │   • memory_vectors   (Layer 2/3)  │   episodes + learning = PER CAMPAIGN
                └──────────────────────────────────┘
```

---

## 3. The two moments that touch storage

### A) `start_call()` — READ (no LLM)
```
start_call(caller_id, campaign)
   │
   ├─ Layer 1: new empty RAM buffer
   └─ Layer 2: load this caller's profile (SHARED)        ── memory_profiles
              load this campaign's episodic store          ── memory_vectors
   →  system_prompt = profile + get_case("kyc") + recall_episodes(k=2)
```

### B) `end_call()` — WRITE (exactly 1 LLM call)
```
end_call(call)
   │
   ├─ transcript = Layer 1 buffer
   ├─ extract_call_record(transcript)        ◄── ONE structured Groq call
   │      → CallRecord{ summary, outcome, facts }
   ├─ Layer 2: update_profile(**facts)        (typed, upsert by key)
   ├─ Layer 2: add_episode(redacted summary)  (importance from outcome)
   └─ Layer 3: winning_scripts OR failure_patterns   (per campaign, redacted)
```

---

## 4. During a turn — what happens (KYC, with function-calling)

```
caller speaks  ─►  Layer 1 records it (free)
                   │
                   ▼
            LLM decides: call update_kyc tool?
                   │ yes
                   ▼
            update_kyc(fields={"pan_no": "ABCDE1234F"})
                   │
                   ├─ VALIDATE (regex)         reject malformed → agent re-asks
                   ├─ ENCRYPT PII fields       (Fernet, at rest)
                   ├─ AUDIT corrections        (encrypted from→to)
                   └─ set_case("kyc", ...)     merge by key (last-write-wins)
                   │
                   ▼
            agent speaks reply  ─►  Layer 1 records it (free)
```
No LLM summary/extraction here — that's deferred to `end_call`. The only LLM cost
mid-call is the agent's own reasoning/tool-use, which you'd pay for in any agent.

---

## 5. The multi-call resume — proven by the demo

Real output (Sheshpal, across two calls):

```
call 1 (busy):
  caller: I'm driving, please call me later.
  agent : I'll call you later to continue with the KYC verification.
  → no docs given → nothing structured saved (case stays empty)

call 2 (resumes, gives Aadhaar, then PAN + name fix):
  caller: My Aadhaar is 1234-5678-9012. I don't have my PAN right now.
  agent : Noted your Aadhaar; we're still missing your PAN.        ← knows what's missing
  caller: Found my PAN, ABCDE1234F. Also my name is Sheshpal Singh, please correct it.
  agent : Updated your name and noted your PAN — we have everything now.
```

Final stored case (PII encrypted at rest):
```
SHESHPAL: collected=[Aadhaar, PAN], missing=[],
          fields={ aadhaar_no: gAAAA…(encrypted),
                   pan_no:     gAAAA…(encrypted),
                   name: "Sheshpal Singh" },
          stage=docs_collected
```

Mid-call correction also proven (Ravi corrected his passport number):
```
RAVI: corrections=[{ field: passport_no,
                     from: gAAAA…(encrypted old),
                     to:   gAAAA…(encrypted new) }]
```

And isolation — three callers, three completely separate cases, no overlap.

---

## 6. Why the prompt doesn't explode

```
Past calls for this caller:  [call#1] [call#2] … [call#10]
                                  │       │           │
                                  └───────┴─── summarised ──┐
                                                            ▼
  System prompt carries:   profile  +  resumed case  +  ≤2 summary lines
                           └────────────── FIXED, SMALL ──────────────┘
                           (NOT 10 full transcripts)

  During the call:  + live turns  →  cap with Layer 1 recent_messages(max_tokens)
```

- Memory adds a **fixed, small** amount (it stores summaries, not transcripts).
- The live conversation grows within one call (normal LLM chat) — bound it with
  Layer 1's token-budgeted window.
- Next call starts fresh: only the summary carries over, never the full prior transcript.

---

## 7. Cost & latency at a glance

| Moment | LLM calls | Latency |
|--------|-----------|---------|
| `start_call` (fetch + build prompt) | 0 | ~ms (RAM + 1 vector read) |
| each turn (record) | 0 (memory) | ~0 |
| optional mid-call retrieval | 0 | ~34 ms (KB, no rerank) |
| `end_call` (summary+outcome+facts) | **1** | ~0.5–1 s (after the call) |

---

## 8. Run it yourself

```bash
python3 lang_memory/kyc_demo.py        # multi-call resume + corrections + PII + isolation
pytest lang_memory/tests/test_api.py   # all REST endpoints (FastAPI TestClient)
pytest lang_memory/tests               # full offline suite (layers, store, isolation, PII, campaigns, API)
```
