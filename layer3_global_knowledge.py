"""
layer3_global_knowledge.py
──────────────────────────
LAYER 3 — GLOBAL KNOWLEDGE (shared across ALL callers, learned from every call).

This layer turns finished calls into reusable knowledge and serves knowledge
back to the agent:

  • call summary    — LLM condenses the transcript into a few lines
  • outcome labeller — LLM tags the call (booked / callback / not_interested /
                       failed / info_only)
  • memory writer    — the bridge: takes a finished call, summarises + labels it,
                       writes the summary into the caller's episodic memory
                       (layer 2) and routes it to the right global store below
  • failure patterns — vector store of how/why calls go wrong
  • winning scripts  — vector store of phrasings that led to good outcomes
  • rag knowledgebase — vector store of product/FAQ facts the agent can quote

LLM calls use Groq via LangChain; the vector stores use the shared FAISS helper.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from .config import get_llm
from .vectorstore import make_store
from .crypto import redact_pii
from .extraction import extract_call_record, found_facts
from .layer2_calling_memory import _safe

# Outcomes are a fixed, small set so they're easy to count and route on.
OUTCOMES = ["booked", "callback", "not_interested", "info_only", "failed"]
GOOD_OUTCOMES = {"booked", "callback"}  # worth saving as a winning script


class GlobalKnowledge:
    def __init__(self, campaign_id: str = "default"):
        # "Global" = shared across all CALLERS, but scoped PER CAMPAIGN: one campaign's
        # winning scripts / failure patterns / KB never leak into another's. (This is
        # conversation-derived learning, so it stays campaign-private — unlike the profile.)
        cid = _safe(campaign_id)
        self.campaign_id = cid
        self.failure_patterns = make_store(f"failure_patterns_{cid}")
        self.winning_scripts  = make_store(f"winning_scripts_{cid}")
        self.knowledge_base   = make_store(f"rag_knowledgebase_{cid}")

    # ── call summary ─────────────────────────────────────────────────────────
    def summarize(self, transcript: str) -> str:
        msgs = [
            SystemMessage(content=(
                "You summarise a phone call between a sales/support agent and a "
                "caller in 2-3 short sentences: what the caller wanted, what was "
                "agreed, and any follow-up. Be factual, no preamble."
            )),
            HumanMessage(content=transcript),
        ]
        return get_llm().invoke(msgs).content.strip()

    # ── outcome labeller ─────────────────────────────────────────────────────
    def label_outcome(self, transcript: str) -> str:
        msgs = [
            SystemMessage(content=(
                "Classify the call outcome as EXACTLY one of: "
                + ", ".join(OUTCOMES) + ". Reply with only that one word."
            )),
            HumanMessage(content=transcript),
        ]
        reply = get_llm().invoke(msgs).content.strip().lower()
        # Be forgiving: match the first known outcome that appears in the reply.
        for o in OUTCOMES:
            if o in reply:
                return o
        return "failed"

    # ── memory writer (the bridge between a finished call and lasting memory) ──
    def write_call(self, calling_memory, transcript: str) -> dict:
        """
        Persist a finished call. ONE structured Groq call extracts summary + outcome +
        typed caller facts; we then upsert the facts into the profile, store the summary
        in the caller's episodic memory, and route the call to the winning/failure global
        stores. Returns {"summary", "outcome"} for logging.

        Cost: exactly 1 Groq call, at call END — ZERO per turn during the live call.
        """
        record = extract_call_record(transcript)              # one structured call
        summary = redact_pii(record.summary)                  # no raw document numbers in stored text
        outcome = record.outcome
        meta = {"caller_id": calling_memory.caller_id, "outcome": outcome}

        # 0) typed facts → the caller's profile (upsert by key; blanks are skipped)
        facts = found_facts(record.facts)
        if facts:
            calling_memory.update_profile(**facts)

        # 1) personal episodic memory (layer 2)
        calling_memory.add_episode(summary, meta)

        # 2) global learning (layer 3) — store the redacted transcript, not the raw one
        if outcome in GOOD_OUTCOMES:
            self.winning_scripts.add(redact_pii(transcript), meta)
        elif outcome in {"not_interested", "failed"}:
            self.failure_patterns.add(summary, meta)

        return {"summary": summary, "outcome": outcome}

    # ── retrieval the live agent uses ────────────────────────────────────────
    # rerank=True is most accurate; pass rerank=False on the hot path for speed
    # (skips the cross-encoder → ~38 ms instead of ~90 ms).
    def recall_winning_scripts(self, query: str, k: int = 3, rerank: bool = True):
        return self.winning_scripts.recall(query, k=k, rerank=rerank)

    def recall_failure_patterns(self, query: str, k: int = 3, rerank: bool = True):
        return self.failure_patterns.recall(query, k=k, rerank=rerank)

    # ── the self-learning feedback: what worked + what to avoid ───────────────
    # This is the READ side of the global learning loop. end_call() writes winning
    # scripts and failure patterns; this surfaces BOTH for the current situation so
    # the agent repeats what worked and dodges past mistakes.
    def guidance(self, query: str, k: int = 1, rerank: bool = True) -> str:
        wins = self.recall_winning_scripts(query, k=k, rerank=rerank)
        fails = self.recall_failure_patterns(query, k=k, rerank=rerank)
        blocks = []
        if wins:
            blocks.append("What worked before:\n" + "\n".join(f"- {d.page_content}" for d in wins))
        if fails:
            blocks.append("What to avoid (past failures):\n" + "\n".join(f"- {d.page_content}" for d in fails))
        return "\n\n".join(blocks)

    # ── RAG knowledge base ───────────────────────────────────────────────────
    def add_knowledge(self, text: str, metadata: dict | None = None):
        self.knowledge_base.add(text, metadata)

    def kb_search(self, query: str, k: int = 3, rerank: bool = True):
        return self.knowledge_base.recall(query, k=k, rerank=rerank)
