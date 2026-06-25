"""trim_history: bound a long call's prompt without splitting a tool-call sequence."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from lang_memory import trim_history


def _two_tool_turns():
    return [
        SystemMessage(content="SYS"),
        HumanMessage(content="turn1 " + "a" * 3000),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="saved", tool_call_id="1"),
        AIMessage(content="ok1"),
        HumanMessage(content="turn2 " + "b" * 3000),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]),
        ToolMessage(content="saved", tool_call_id="2"),
        AIMessage(content="ok2"),
        HumanMessage(content="turn3 current"),
    ]


def test_noop_when_under_budget():
    h = _two_tool_turns()
    assert trim_history(h, max_chars=100_000) == h     # nothing dropped


def test_drops_oldest_whole_turn_and_keeps_integrity():
    h = _two_tool_turns()
    out = trim_history(h, max_chars=2500)              # tight → must drop the oldest turn
    assert len(out) < len(h)
    assert isinstance(out[0], SystemMessage)           # system kept
    assert isinstance(out[1], HumanMessage)            # starts at a caller turn
    assert out[-1].content == "turn3 current"          # current turn preserved
    assert not any("turn1" in (m.content or "") for m in out)   # oldest turn gone
    # every ToolMessage still follows its AI tool-call (no orphans)
    for i, m in enumerate(out):
        if isinstance(m, ToolMessage):
            assert isinstance(out[i - 1], AIMessage) and out[i - 1].tool_calls
