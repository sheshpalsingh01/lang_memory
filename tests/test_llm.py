"""LLM-backed tests (Layer 3 summary/label + end-to-end write). Opt-in.

These hit the Groq API, so they're skipped by default. Run them with:
    RUN_LLM_TESTS=1 pytest lang_memory/tests/test_llm.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LLM_TESTS"), reason="set RUN_LLM_TESTS=1 to run LLM tests"
)

from lang_memory import VoiceAgentMemory
from lang_memory.layer3_global_knowledge import GlobalKnowledge, OUTCOMES
from lang_memory.extraction import extract_call_record

TRANSCRIPT = "Caller: Please book a demo for Monday.\nAgent: Booked Monday 3 PM, confirmation by SMS."


def test_summarize_returns_text():
    s = GlobalKnowledge().summarize(TRANSCRIPT)
    assert isinstance(s, str) and len(s) > 10


def test_label_outcome_is_valid():
    assert GlobalKnowledge().label_outcome(TRANSCRIPT) in OUTCOMES


def test_extract_call_record_one_call():
    rec = extract_call_record(TRANSCRIPT)
    assert rec.summary and rec.outcome in OUTCOMES   # outcome is validated by the Literal type


def test_end_call_persists_episode():
    m = VoiceAgentMemory()
    c = m.start_call("+91-test")
    c.add_caller("Please book a demo for Monday.")
    c.add_agent("Booked Monday 3 PM, confirmation by SMS.")
    result = m.end_call(c)
    assert result["outcome"] in OUTCOMES
    # the call must be recallable from the caller's episodic memory afterwards
    assert m.start_call("+91-test").calling.recall_episodes("demo", k=1)


def test_auto_reflect_runs_every_n_calls():
    m = VoiceAgentMemory(reflect_every=2)          # reflect after every 2 calls
    cid = "+91-reflect"
    for text in ("Always contact me by SMS, I never pick up calls.",
                 "Reminder: SMS only please."):
        c = m.start_call(cid)
        c.add_caller(text); c.add_agent("Noted.")
        m.end_call(c)
    # after 2 calls, reflection should have distilled at least one durable preference
    assert m.start_call(cid).calling.get_preferences()


def test_facts_extracted_and_land_in_profile():
    rec = extract_call_record("Caller: This is Ravi, please call me back after 6pm in Hindi.")
    assert rec.facts.name and "Ravi" in rec.facts.name
    # end_call should upsert the typed facts into the caller's profile (same 1 call)
    m = VoiceAgentMemory()
    c = m.start_call("+91-facts")
    c.add_caller("Hi this is Meera, I prefer Hindi. Please call after 6pm.")
    c.add_agent("Sure Meera, noted.")
    m.end_call(c)
    assert m.start_call("+91-facts").calling.get_profile()  # non-empty: facts persisted
