"""Semi-isolation across campaigns:
  • profile (identity) is SHARED per caller across campaigns
  • episodic history + global learning are PRIVATE per campaign
"""

from lang_memory import VoiceAgentMemory


def test_profile_shared_history_isolated():
    cid = "+91-shared-caller"
    camp_a = VoiceAgentMemory(campaign_id="loans")
    camp_b = VoiceAgentMemory(campaign_id="insurance")

    # caller sets identity + a call in campaign A
    a = camp_a.start_call(cid)
    a.calling.update_profile(name="Ravi", language="hi")
    a.calling.add_episode("discussed a home loan top-up", {"outcome": "callback"})

    # campaign B sees the SHARED profile...
    b = camp_b.start_call(cid)
    assert b.calling.get_profile() == {"name": "Ravi", "language": "hi"}

    # ...but NOT campaign A's conversation history
    assert b.calling.recall_episodes("loan") == []
    # while campaign A still has its own history
    assert camp_a.start_call(cid).calling.recall_episodes("loan")


def test_global_learning_isolated_per_campaign():
    a = VoiceAgentMemory(campaign_id="loans")
    b = VoiceAgentMemory(campaign_id="insurance")
    a.knowledge.add_knowledge("EMI starts the month after disbursal")
    # campaign B's KB must not contain campaign A's knowledge
    assert b.knowledge.kb_search("EMI", rerank=False) == []
    assert a.knowledge.kb_search("EMI", rerank=False)        # A still finds it
