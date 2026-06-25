"""
pipecat_integration.py
──────────────────────
Drop-in Pipecat processor that wires the 3-layer memory into a LIVE voice call.

    pip install pipecat-ai          # this file needs Pipecat installed

`LangMemoryProcessor` is a normal `FrameProcessor` you place in the pipeline. It
taps the frame stream and, frame by frame:

  • StartFrame                       → open the call (load the caller's layer-2 memory)
  • TranscriptionFrame  (caller)     → record the turn; optionally inject live RAG
  • TTSTextFrame        (bot)        → record what the bot actually said
  • End / Cancel Frame               → summarise + label + persist the call (layer-3 writer)

Where it goes in the pipeline — right after STT, before the LLM aggregator, and the
same instance sees the bot's TTS text on the way out:

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.services.openai import OpenAILLMContext
    from lang_memory import VoiceAgentMemory
    from lang_memory.pipecat_integration import LangMemoryProcessor

    memory   = VoiceAgentMemory(required_slots=["date", "time"])
    context  = OpenAILLMContext()                       # your LLM context
    mem_proc = LangMemoryProcessor(memory, caller_id=caller_phone, context=context)

    # personalise the very first system message from what we already know:
    context.add_message(mem_proc.opening_memory_message())

    pipeline = Pipeline([
        transport.input(),
        stt,
        mem_proc,                       # ← records caller turns, persists on end
        context_aggregator.user(),
        llm,
        tts,                            # emits TTSTextFrame → mem_proc records bot turns
        transport.output(),
        context_aggregator.assistant(),
    ])

Blocking work (Groq summary, embeddings, cross-encoder) is run via asyncio.to_thread
so it never stalls the real-time audio loop. NLU (intent/slots) stays in your app —
reach `mem_proc.session` to call `set_intent` / `set_slot` from your own logic.

COST MODEL (simple and predictable — no hidden charges):
  • Call start : ONE compact memory message is added to the prompt (profile +
                 preferences + up to 2 recent call summaries). It stays the same
                 for the whole call, so the prompt does NOT grow turn by turn.
  • During call: nothing is added to the prompt by default. Turns are only recorded
                 locally (free) for the end-of-call summary.
  • Call end   : ONE Groq call summarises + labels the call, then it is saved.
  So per call the only extra LLM cost is that single end-of-call call.

  Live RAG (look up a fact mid-call and inject it) is OFF by default. Turn it on with
  inject_rag=True only if you want it — then one small grounding line is REPLACED each
  turn (it never piles up).
"""

import asyncio

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .voice_agent_memory import VoiceAgentMemory, CallSession
from .prompt import scalar_fields


class LangMemoryProcessor(FrameProcessor):
    """Bridges Pipecat's frame stream and the 3-layer VoiceAgentMemory."""

    def __init__(self, memory: VoiceAgentMemory, caller_id: str,
                 context=None, inject_rag: bool = False, retrieve_k: int = 3,
                 fast: bool = True):
        super().__init__()
        self.memory = memory
        self.caller_id = caller_id
        self.context = context              # optional LLM context to inject live RAG into
        self.inject_rag = inject_rag
        self.retrieve_k = retrieve_k
        # fast=True → hot-path retrieval is KB-only with NO cross-encoder rerank
        # (~38 ms/turn). The caller's episodes + winning scripts don't change during
        # a call, so they are loaded ONCE at call start, not re-fetched every turn.
        self.fast = fast
        self.session: CallSession | None = None
        self._opening_injected = False
        self._last_grounding = None     # the single rolling RAG message (see _inject_grounding)
        self._persisted = False

    # ── helpers the app calls when building the pipeline ─────────────────────
    def _ensure_session(self) -> CallSession:
        if self.session is None:
            self.session = self.memory.start_call(self.caller_id)
        return self.session

    def opening_memory_message(self) -> dict:
        """A small system message with the caller's profile, preferences and up to 2
        recent/important past calls. Added ONCE at call start; it stays fixed all call,
        so the prompt never grows turn by turn. Keep it short to keep the prompt cheap."""
        s = self._ensure_session()
        # scalar identity fields only — never dump nested case dicts / encrypted PII blobs
        profile = scalar_fields(s.calling.get_profile())
        prefs = s.calling.get_preferences()
        episodes = s.calling.recall_episodes("summary of our previous conversations", k=2)
        lines = []
        if profile:
            lines.append("Caller profile: " + ", ".join(f"{k}={v}" for k, v in profile.items()))
        if prefs:
            lines.append("Known preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items()))
        if episodes:
            lines.append("Relevant past calls:")
            lines += [f"- {d.page_content}" for d in episodes]
        content = "\n".join(lines) if lines else "No prior history for this caller."
        return {"role": "system", "content": content}

    async def retrieve(self, query: str) -> str:
        """
        HOT-PATH grounding for the current caller turn. In fast mode this is
        KB-only with no rerank (~38 ms) — episodes/scripts were already loaded at
        call start. In full mode it also pulls reranked episodes + a winning script.
        Runs off the event loop.
        """
        def _work():
            kb = self.memory.knowledge.kb_search(query, k=self.retrieve_k, rerank=not self.fast)
            blocks = []
            if kb:
                blocks.append("[Knowledge]\n" + "\n".join(d.page_content for d in kb))
            if not self.fast:
                eps = self._ensure_session().calling.recall_episodes(query, k=self.retrieve_k)
                if eps:
                    blocks.append("[This caller's history]\n" + "\n".join(d.page_content for d in eps))
                # global self-learning: surfaces BOTH winning scripts and failure patterns
                guidance = self.memory.knowledge.guidance(query, k=1)
                if guidance:
                    blocks.append("[Lessons from past calls]\n" + guidance)
            return "\n\n".join(blocks)
        return await asyncio.to_thread(_work)

    # ── the Pipecat hook ─────────────────────────────────────────────────────
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # One-time static load: caller profile + preferences + past calls.
            # These don't change during the call, so they go in ONCE here, off the
            # per-turn hot path.
            self._ensure_session()
            if self.context is not None and not self._opening_injected and hasattr(self.context, "add_message"):
                await asyncio.to_thread(self.context.add_message, self.opening_memory_message())
                self._opening_injected = True

        elif isinstance(frame, TranscriptionFrame) and frame.text:
            # caller said something (final transcript)
            self._ensure_session().add_caller(frame.text)
            if self.inject_rag and self.context is not None:
                grounding = await self.retrieve(frame.text)
                if grounding:
                    self._inject_grounding(grounding)

        elif isinstance(frame, TTSTextFrame) and frame.text:
            # the bot's spoken words
            self._ensure_session().add_agent(frame.text)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._persist()

        await self.push_frame(frame, direction)

    def _inject_grounding(self, text: str):
        """
        Keep exactly ONE rolling RAG message in the context. We REMOVE last turn's
        grounding before adding this turn's, so the prompt does NOT grow with the
        number of turns — per-turn injected tokens stay constant.
        """
        if not hasattr(self.context, "add_message"):
            return
        msgs = getattr(self.context, "messages", None)
        if isinstance(msgs, list) and self._last_grounding in msgs:
            msgs.remove(self._last_grounding)
        msg = {"role": "system", "content": "Use if relevant:\n" + text}
        self.context.add_message(msg)
        self._last_grounding = msg

    async def _persist(self):
        """Summarise + label + write the call to layers 2 & 3. Runs once, off-loop."""
        if self._persisted or self.session is None:
            return
        self._persisted = True
        await asyncio.to_thread(self.memory.end_call, self.session)
