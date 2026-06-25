"""
extraction.py
─────────────
Typed extraction (best practice: structure what you remember, in ONE call).

At call END a single structured Groq call reads the transcript once and returns
everything we persist — the summary, the outcome, and durable caller facts — as a
validated Pydantic object. No second pass, no regex parsing of the reply.

Facts are scalar on purpose: a re-extracted value overwrites the old one by key
(deterministic last-write-wins) — no merge logic, no LLM dedup needed.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .config import get_llm

# kept in sync with layer3.OUTCOMES (duplicated here only to avoid a circular import)
_OUTCOMES = Literal["booked", "callback", "not_interested", "info_only", "failed"]


class CallFacts(BaseModel):
    """Durable, non-sensitive facts about the caller worth remembering for next time.
    (Document numbers are NOT here — those go through the KYC tool, encrypted.)"""
    name: str | None = Field(None, description="the caller's name, if they gave it")
    preferred_language: str | None = Field(None, description="language they prefer, e.g. 'Hindi'")
    preferred_callback_time: str | None = Field(None, description="when to call back, e.g. 'after 6pm'")
    do_not_call: bool = Field(False, description="true only if the caller asked not to be contacted again")


class CallRecord(BaseModel):
    """Everything we extract from a finished call — produced in ONE structured call."""
    summary: str = Field(description="2-3 factual sentences: what the caller wanted, what was agreed, any follow-up")
    outcome: _OUTCOMES = Field(description="the call outcome")
    facts: CallFacts = Field(default_factory=CallFacts, description="durable facts about the caller")


def extract_call_record(transcript: str) -> CallRecord:
    """ONE structured Groq call at call-end → summary + outcome + typed facts.
    Falls back to a basic record if the LLM is down (never crash end_call)."""
    try:
        return get_llm().with_structured_output(CallRecord).invoke(
            "Summarise this phone call, classify its outcome, and extract durable facts "
            "about the caller. Leave a fact blank if the call didn't reveal it.\n\n" + transcript
        )
    except Exception:
        return CallRecord(summary=transcript[:300], outcome="failed")


def found_facts(facts: CallFacts) -> dict:
    """Only the fields the call actually revealed — so we never blank a known value."""
    return {k: v for k, v in facts.model_dump().items() if v not in (None, "", False)}
