"""
config.py
─────────
ONE place where every setting and every shared heavy object lives.

The two heavy objects are:
  • the embedding model  (loads once, ~100 MB, used by layers 2 and 3)
  • the Groq chat model   (used by layer 3 for summaries / outcome labels)

We build each one LAZILY and CACHE it, so importing the package is cheap and
the embedding model is only loaded the first time something needs a vector.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # read final_RAG/.env style files if present


class Settings:
    # ── Groq LLM (layer 3: summary + outcome labelling) ───────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # ── Embedding model (runs locally, no API key) ────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    # ── Cross-encoder reranker (runs locally) — the precision booster ──────────
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Storage ───────────────────────────────────────────────────────────────
    # File backend by default (great for dev/tests). Set DATABASE_URL to switch the
    # whole package to Postgres + pgvector (concurrency + scale) with no code changes.
    DATA_DIR: str     = os.getenv("LANG_MEMORY_DIR", "./data/lang_memory")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")          # e.g. postgresql://user:pw@host/db
    EMBED_DIM: int    = int(os.getenv("EMBED_DIM", "768"))     # bge-base-en-v1.5 = 768

    # ── PII encryption (production: set a real Fernet key) ────────────────────
    ENC_KEY: str = os.getenv("LANG_MEMORY_ENC_KEY", "")        # empty → dev fallback in crypto.py
    RETENTION_DAYS: int = int(os.getenv("RETENTION_DAYS", "0"))  # 0 = keep forever


settings = Settings()


@lru_cache(maxsize=1)
def get_embeddings():
    """Shared embedding model — built once, reused everywhere."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm():
    """Shared Groq chat model for layer 3. Raises if no API key is set."""
    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set — layer 3 (summary / outcome) needs it.\n"
            "Add it to your .env. Get a free key at https://console.groq.com"
        )
    from langchain_groq import ChatGroq

    # temperature=0 → the same call summary / outcome label every time.
    return ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY,
                    temperature=0)
