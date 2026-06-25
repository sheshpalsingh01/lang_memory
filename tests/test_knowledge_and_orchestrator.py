"""Layer 3 (non-LLM parts) + the VoiceAgentMemory orchestrator."""

from lang_memory import VoiceAgentMemory
from lang_memory.layer3_global_knowledge import GlobalKnowledge


# ── Layer 3: RAG KB + the self-learning guidance loop ────────────────────────
def test_kb_add_and_search():
    g = GlobalKnowledge()
    g.add_knowledge("refunds are processed within 5 business days", {"k": "refund"})
    top = g.kb_search("how long does a refund take", k=1)
    assert top and "5 business days" in top[0].page_content


def test_guidance_combines_winning_and_failure():
    g = GlobalKnowledge()
    g.winning_scripts.add("offered the discount and the customer booked", {"outcome": "booked"})
    g.failure_patterns.add("pushed too hard and the customer hung up", {"outcome": "failed"})
    out = g.guidance("how to handle a hesitant customer", k=1)
    assert "What worked before" in out
    assert "What to avoid" in out


def test_guidance_empty_when_nothing_learned():
    assert GlobalKnowledge().guidance("anything") == ""


# ── Orchestrator ─────────────────────────────────────────────────────────────
def test_start_call_sets_up_layers():
    c = VoiceAgentMemory(["date"]).start_call("x")
    assert c.active.caller_id == "x"
    assert "date" in c.active.slots


def test_callsession_passthroughs():
    c = VoiceAgentMemory(["date"]).start_call("x")
    c.set_intent("book")
    c.update_entities(city="Pune")
    c.set_slot("date", "2026-07-01")
    c.add_caller("hi")
    assert c.active.intent == "book"
    assert c.active.entities == {"city": "Pune"}
    assert c.key_points()["slots"] == {"date": "2026-07-01"}


def test_context_for_opening():
    m = VoiceAgentMemory()
    c = m.start_call("x")
    c.calling.update_profile(name="A")
    assert m.context_for_opening(c)["profile"] == {"name": "A"}


def test_end_call_empty_transcript_no_llm():
    """An empty call returns failed WITHOUT calling the LLM (the guard)."""
    m = VoiceAgentMemory()
    c = m.start_call("x")
    assert m.end_call(c) == {"summary": "", "outcome": "failed"}
