"""
prompt.py
─────────
ONE safe way to inject memory into a system prompt.

Why this exists: dumping `get_profile()` straight into the prompt is unsafe — the
profile also holds nested "case" objects (e.g. the KYC case) whose fields are
ENCRYPTED ciphertext. That would leak useless ciphertext into the prompt and
duplicate data. This builder includes only the safe, human-readable SCALAR identity
fields (name, language, …) and the recalled past-call summaries.

  system = build_system_prompt(call, "You are a KYC agent for XYZ bank.")

Domain-specific structured state (like the masked KYC case) is intentionally NOT
added here — render it yourself with masking, the way kyc_demo.system_prompt does.

Also provides trim_history(): bound the LLM message list on a LONG call so the prompt
stays capped, without ever splitting a tool-call sequence.
"""

from langchain_core.messages import HumanMessage


def trim_history(history, max_chars: int = 8000):
    """Keep the system message + the most recent turns within a budget, so a LONG call's
    prompt stays bounded. Trims whole turns only — the kept window starts at a caller
    (Human) message, so an AI tool-call + its ToolMessage are never separated. Call this
    only at a turn boundary (no half-finished tool sequence)."""
    if len(history) <= 2:
        return history
    system, rest = history[:1], history[1:]
    kept, used = [], 0
    for m in reversed(rest):                       # newest first, until the budget is hit
        used += len(m.content or "") if isinstance(m.content, str) else 0
        kept.append(m)
        if used > max_chars:
            break
    kept.reverse()
    while kept and not isinstance(kept[0], HumanMessage):   # don't start mid-tool-sequence
        kept.pop(0)
    return system + kept


def scalar_fields(d: dict) -> dict:
    """Keep only simple, printable identity fields — drops nested case dicts/lists (which
    hold encrypted PII blobs) and empty values. Shared by every place that exposes a
    profile (system prompt, Pipecat opening message, the /callers API)."""
    return {k: v for k, v in d.items()
            if isinstance(v, (str, int, float, bool)) and v not in ("", None)}


def _kv(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def build_system_prompt(call, base_prompt: str, k: int = 2,
                        recall_query: str = "our previous conversations") -> str:
    """base prompt + caller identity + preferences + up to k past-call summaries.
    Built ONCE at call start; safe to send to the LLM (no PII ciphertext)."""
    profile = scalar_fields(call.calling.get_profile())
    prefs   = scalar_fields(call.calling.get_preferences())
    past    = call.calling.recall_episodes(recall_query, k=k)

    lines = [base_prompt]
    if profile:
        lines.append("Caller: " + _kv(profile))
    if prefs:
        lines.append("Preferences: " + _kv(prefs))
    if past:
        lines.append("Past calls:\n" + "\n".join(f"- {d.page_content}" for d in past))
    return "\n".join(lines)
