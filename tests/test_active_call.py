"""Layer 1 — ActiveCallMemory (in-RAM, no models, fast)."""

from lang_memory.layer1_active_call import ActiveCallMemory


def test_records_turns():
    a = ActiveCallMemory("c")
    a.add_caller("hi")
    a.add_agent("hello")
    assert [m.content for m in a.messages()] == ["hi", "hello"]


def test_empty_and_none_skipped():
    a = ActiveCallMemory("c")
    a.add_caller("")
    a.add_caller("   ")
    a.add_agent(None)
    assert a.messages() == []


def test_strips_whitespace():
    a = ActiveCallMemory("c")
    a.add_caller("  hi  ")
    assert a.messages()[0].content == "hi"


def test_recent_messages_is_a_window():
    a = ActiveCallMemory("c", max_context_tokens=5)
    for i in range(10):
        a.add_caller(f"message number {i} here please")
    recent = a.recent_messages()
    assert len(a.messages()) == 10          # full history kept
    assert len(recent) < 10                  # prompt window is smaller
    assert recent[-1] is a.messages()[-1]    # keeps the newest turn


def test_recent_keeps_at_least_one_oversized():
    a = ActiveCallMemory("c", max_context_tokens=1)
    a.add_caller("a very long message that far exceeds the tiny token budget")
    assert len(a.recent_messages()) == 1


def test_slots_complete_and_missing():
    a = ActiveCallMemory("c", ["date", "time"])
    assert not a.is_complete()
    a.set_slot("date", "2026-07-01")
    assert a.missing_slots() == ["time"]
    a.set_slot("time", "15:00")
    assert a.is_complete()


def test_key_points_structure():
    a = ActiveCallMemory("c", ["date"])
    a.set_intent("book")
    a.update_entities(city="Pune")
    a.set_slot("date", "2026-07-01")
    kp = a.key_points()
    assert kp["intent"] == "book"
    assert kp["slots"] == {"date": "2026-07-01"}
    assert kp["entities"] == {"city": "Pune"}


def test_state_summary():
    a = ActiveCallMemory("c", ["date"])
    a.set_intent("book")
    assert "book" in a.state_summary()
    assert ActiveCallMemory("c2").state_summary() == "no key points yet"


def test_transcript_format():
    a = ActiveCallMemory("c")
    a.add_caller("hi")
    a.add_agent("yo")
    assert a.transcript() == "Caller: hi\nAgent: yo"
