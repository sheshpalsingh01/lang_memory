"""
lang_memory — 3-layer memory for an AI voice calling agent, built on LangChain.

  Layer 1  ActiveCallMemory   — conversation buffer, intent, entities, slots
  Layer 2  CallingMemory      — profile store, episodic memory, preference learning
  Layer 3  GlobalKnowledge    — call summary, outcome labeller, memory writer,
                                failure patterns, winning scripts, RAG knowledge base

Use VoiceAgentMemory to drive all three across a call's lifecycle.
"""

from .layer1_active_call import ActiveCallMemory
from .layer2_calling_memory import CallingMemory
from .layer3_global_knowledge import GlobalKnowledge
from .voice_agent_memory import VoiceAgentMemory, CallSession
from .prompt import build_system_prompt, trim_history

__all__ = [
    "ActiveCallMemory",
    "CallingMemory",
    "GlobalKnowledge",
    "VoiceAgentMemory",
    "CallSession",
    "build_system_prompt",
    "trim_history",
]
