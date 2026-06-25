"""Layer 2 — CallingMemory: profile, preferences, case state, episodic, isolation."""

import glob

from lang_memory.layer2_calling_memory import CallingMemory, _safe


def test_profile_persists():
    CallingMemory("c").update_profile(name="Ravi")
    assert CallingMemory("c").get_profile() == {"name": "Ravi"}


def test_preferences_persist():
    cm = CallingMemory("c")
    cm.learn_preference("contact_channel", "sms")
    assert cm.get_preferences() == {"contact_channel": "sms"}


def test_set_case_merges():
    cm = CallingMemory("c")
    cm.set_case("kyc", stage="a", collected=["aadhaar"])
    cm.set_case("kyc", stage="b")                 # partial update
    case = cm.get_case("kyc")
    assert case["stage"] == "b"                    # updated
    assert case["collected"] == ["aadhaar"]        # merge kept the rest


def test_get_case_default_empty():
    assert CallingMemory("c").get_case("kyc") == {}


def test_caller_id_sanitized():
    assert _safe("+91 11-22") == "_91_11-22"
    assert _safe("") == "anonymous"


def test_atomic_write_leaves_no_tmp(tmp_path):
    CallingMemory("c").update_profile(x=1)
    assert glob.glob(str(tmp_path) + "/**/*.tmp", recursive=True) == []


def test_episodic_add_and_recall():
    cm = CallingMemory("c")
    cm.add_episode("the caller booked a product demo", {"outcome": "booked"})
    top = cm.recall_episodes("demo booking", k=1)
    assert top and "demo" in top[0].page_content


def test_episodic_empty():
    assert CallingMemory("c").recall_episodes("anything") == []


def test_callers_are_isolated():
    CallingMemory("alice").add_episode("alice booked a demo", {})
    CallingMemory("bob").add_episode("bob asked for a refund", {})
    alice_hits = CallingMemory("alice").recall_episodes("refund", k=5)
    assert all("bob" not in d.page_content for d in alice_hits)
    assert CallingMemory("alice").episodes.name != CallingMemory("bob").episodes.name
