"""
kyc_demo.py
───────────
A REAL function-calling KYC agent on top of the 3-layer memory, showing that:

  • the agent RESUMES each caller's case across calls (memory),
  • the LLM EXTRACTS document info from free speech and writes it (function-calling),
  • the conversation is DIFFERENT for every caller — nothing is scripted into the agent;
    it just reacts to what each person says,
  • each caller's memory is ISOLATED (one person never sees another's case).

Memory stores; the LLM extracts. Uses the Groq key in config. The agent runs on a
stronger model (llama-3.3-70b-versatile) for clean tool-use + spoken replies; the
memory's own summaries stay on the cheaper config default.

Run:  python3 lang_memory/kyc_demo.py
"""

import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from lang_memory import VoiceAgentMemory
from lang_memory.config import settings
from lang_memory.crypto import encrypt, decrypt
from lang_memory.prompt import trim_history

memory = VoiceAgentMemory()

# These document numbers are PII — encrypt their values at rest, decrypt only for a
# verified caller. (Free-text summaries are redacted automatically by layer 3.)
PII_FIELDS = {"pan_no", "aadhaar_no", "passport_no"}
def _enc_fields(f): return {k: (encrypt(v) if k in PII_FIELDS else v) for k, v in f.items()}
def _dec_fields(f): return {k: (decrypt(v) if k in PII_FIELDS else v) for k, v in f.items()}
def _mask_tail(f):
    # Unverified caller: reveal only the LAST 4 chars so the agent can ask them to
    # confirm a stored detail (bank-style), without exposing the full number.
    out = {}
    for k, v in _dec_fields(f).items():
        v = str(v)
        out[k] = ("*" * (len(v) - 4) + v[-4:]) if len(v) > 4 else "****"
    return out
_active = {}                                  # the call currently being handled (for the tool)

AGENT_MODEL = "llama-3.3-70b-versatile"       # stronger model for the live agent

# Edge case: STT mis-hears digits / the model hallucinates. Reject malformed doc
# numbers so a wrong value is never stored — the agent is told to re-ask.
VALIDATORS = {
    "pan_no":      (r"^[A-Z]{5}[0-9]{4}[A-Z]$",     "PAN must be 5 letters, 4 digits, 1 letter"),
    "aadhaar_no":  (r"^[0-9]{12}$",                  "Aadhaar must be exactly 12 digits"),
    "passport_no": (r"^[A-Z][0-9]{7}$",              "passport must be 1 letter + 7 digits"),
}


# ── the tool the LLM calls to write structured KYC state into memory ─────────
@tool
def update_kyc(stage: str = "", collected: list[str] = None, missing: list[str] = None,
               fields: dict = None, pending: str = "") -> str:
    """Record or correct the caller's KYC case.
    stage: pending|in_progress|docs_collected|confirmed.
    collected/missing: document names. fields: extracted values e.g. {"pan_no":"ABCDE1234F"}.
    pending: the next follow-up. Pass only what you learned this turn."""
    call = _active["call"]
    cur = call.calling.get_case("kyc")
    # Validate document numbers BEFORE storing — a wrong KYC number is a failed
    # verification, not a typo. Reject and make the agent read it back / re-ask.
    for k, v in (fields or {}).items():
        pat = VALIDATORS.get(k)
        if pat and not re.match(pat[0], str(v).upper().replace(" ", "").replace("-", "")):
            return f"REJECTED {k}='{v}' — {pat[1]}. Read it back to the caller and ask them to repeat, then call update_kyc again."
    # Reconcile in CODE so the case is always consistent regardless of the model:
    new_collected = sorted(set(cur.get("collected", [])) | set(collected or []))
    base_missing = missing if missing is not None else cur.get("missing", [])
    new_missing = [m for m in base_missing if m not in new_collected]   # a collected doc is never "missing"
    cur_fields = _dec_fields(cur.get("fields", {}))                    # work in plaintext
    # Audit trail: record any value that changed (KYC compliance — keep what was corrected).
    corrections = list(cur.get("corrections", []))
    for k, v in (fields or {}).items():
        if k in cur_fields and cur_fields[k] != v:
            old, new = (cur_fields[k], v)
            if k in PII_FIELDS:                       # keep the audit trail encrypted too
                old, new = encrypt(old), encrypt(new)
            corrections.append({"field": k, "from": old, "to": new})
    new_fields = {**cur_fields, **(fields or {})}                       # corrections overwrite
    # encrypt PII values before they hit storage
    upd = {"collected": new_collected, "missing": new_missing,
           "fields": _enc_fields(new_fields), "corrections": corrections}
    if stage:   upd["stage"] = stage
    if pending: upd["pending"] = pending
    call.calling.set_case("kyc", **upd)
    return "saved"                                    # don't echo PII back into the chat history


@tool
def verify_identity() -> str:
    """Unlock the caller's stored details. Call ONLY after the caller correctly answers a
    security question (e.g. their date of birth). Never reveal stored details before this."""
    _active["call"].calling.set_case("kyc", identity_verified=True)
    return "identity verified — stored details unlocked"


TOOLS = {"update_kyc": update_kyc, "verify_identity": verify_identity}
llm = ChatGroq(model=AGENT_MODEL, api_key=settings.GROQ_API_KEY, temperature=0).bind_tools(list(TOOLS.values()))


def system_prompt(call) -> str:
    kyc = call.calling.get_case("kyc")
    past = call.calling.recall_episodes("kyc verification", k=2)
    # Edge case: wrong person on a known number. PII is encrypted at rest; decrypt it for
    # the agent ONLY once identity is verified, otherwise mask it (and hide the change history).
    shown = dict(kyc)
    if kyc.get("fields"):
        if kyc.get("identity_verified"):
            shown["fields"] = _dec_fields(kyc["fields"])
        else:
            shown["fields"] = _mask_tail(kyc["fields"])
            shown.pop("corrections", None)
    p = [
        "You are a KYC verification agent for XYZ bank on a phone call. Be brief and natural.",
        "RULES:",
        "- When the caller gives/corrects a document NUMBER, read it back and get a 'yes' before calling update_kyc.",
        "- If a correction is unclear, ASK for the correct value — never blank a field or guess.",
        "- When a previously-missing document is now provided, put it in `collected` and drop it from `missing`.",
        "- Stored numbers are shown masked (e.g. '********9012'). You MAY read out the visible last 4 "
        "digits so the caller can confirm a detail, and repeat them if asked. To reveal or confirm the "
        "FULL number, first verify the caller (ask a security question) and call verify_identity.",
        "- After recording, ALWAYS reply in one short spoken sentence (confirm + next step).",
        "Current KYC case: " + (json.dumps(shown) if shown else "none — start fresh"),
    ]
    if past:
        p.append("Past calls:\n" + "\n".join(f"- {d.page_content}" for d in past))
    return "\n".join(p)


def agent_turn(call, history, user_text):
    """One caller utterance → keep running tool calls until the agent has a spoken reply.
    Bounded loop so a multi-step turn (e.g. verify_identity THEN update_kyc, or a re-ask
    after a rejected value) isn't dropped."""
    call.add_caller(user_text)
    history.append(HumanMessage(content=user_text))
    history[:] = trim_history(history)             # bound the window at this clean boundary
    ai = llm.invoke(history)
    history.append(ai)
    for _ in range(4):                        # cap = safety against a tool-call loop
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            history.append(ToolMessage(content=TOOLS[tc["name"]].invoke(tc["args"]), tool_call_id=tc["id"]))
        ai = llm.invoke(history)
        history.append(ai)
    reply = (ai.content or "").strip() or "Okay, noted."
    call.add_agent(reply)
    return reply


def run_call(caller_id, label, utterances):
    call = memory.start_call(caller_id)
    _active["call"] = call
    history = [SystemMessage(content=system_prompt(call))]
    print(f"\n{'-'*72}\n{label}\n  [resumed case]: {call.calling.get_case('kyc') or 'none'}")
    for u in utterances:
        print(f"  caller: {u}")
        print(f"  agent : {agent_turn(call, history, u)}")
    memory.end_call(call)


# Three DIFFERENT people, three DIFFERENT conversations — the agent only reacts to
# what each one says; nothing about the flow is hard-coded.
SCENARIOS = [
    ("+91-99999-11111", "SHESHPAL — busy first, then docs over two calls, with a correction", [
        ("  call 1 (busy)", ["I'm driving, please call me later."]),
        ("  call 2 (gives Aadhaar, PAN not handy, then finds it + fixes name)", [
            "Okay I have time. My Aadhaar number is 1234-5678-9012. I don't have my PAN on me right now.",
            "Actually I found my PAN, it's ABCDE1234F. Also my name should be Sheshpal Singh, please correct it.",
        ]),
    ]),
    ("+91-99999-22222", "ANITA — everything ready, finishes in one short call", [
        ("  call 1 (all docs at once)", [
            "Hi, I want to finish my KYC quickly. Aadhaar 9876-5432-1098 and PAN PQRST5678U. That's everything.",
        ]),
    ]),
    ("+91-99999-33333", "RAVI — uses passport instead of PAN, corrects a typo", [
        ("  call 1 (passport + a self-correction)", [
            "I'll verify with my passport, number P1234567, and my PAN is ZZZZZ0000A.",
            "Sorry, the passport number is actually P7654321 — I misspoke.",
        ]),
    ]),
]


if __name__ == "__main__":
    for caller_id, who, calls in SCENARIOS:
        print(f"\n{'='*72}\n{who}\n{'='*72}")
        for label, utterances in calls:
            run_call(caller_id, label, utterances)

    print(f"\n{'='*72}\nFINAL CASES — different per person, fully isolated:")
    for caller_id, who, _ in SCENARIOS:
        case = memory.start_call(caller_id).calling.get_case("kyc")
        print(f"  {who.split(' — ')[0]:9s}: {case}")
