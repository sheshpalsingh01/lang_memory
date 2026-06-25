"""
pg_store.py
───────────
Postgres backend — activated when DATABASE_URL is set (otherwise the file backend
in vectorstore.py / layer2 is used). It fixes the file backend's two production
gaps: concurrent callers (row-level, not a shared JSON file) and scale.

Two stores, each with the SAME api as the file versions so nothing else changes:

  • PgHybridStore — vectors. Keeps the hybrid design IN Postgres:
        pgvector (dense)  +  tsvector full-text (sparse)  →  RRF  →  cross-encoder rerank
  • pg_load_profile / pg_save_profile — the per-caller profile/case JSONB row.

Connections are opened per call (autocommit) — safe under concurrency; on localhost
the overhead is sub-10ms, far cheaper than the file race it removes.
"""

import json
from functools import lru_cache

from psycopg_pool import ConnectionPool
from langchain_core.documents import Document

from .config import settings, get_embeddings
from .vectorstore import get_reranker, _rrf


@lru_cache(maxsize=1)
def _get_pool() -> ConnectionPool:
    """One shared pool per process — reuses connections instead of opening one per call
    (removes the connect overhead and the ~100-connection ceiling under concurrency)."""
    return ConnectionPool(settings.DATABASE_URL, min_size=1, max_size=10,
                          kwargs={"autocommit": True}, open=True)


def _conn():
    """Borrow a pooled connection; returned to the pool when the `with` block exits."""
    return _get_pool().connection()


@lru_cache(maxsize=1)
def _init_schema():
    """Create the tables/indexes once per process."""
    with _conn() as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(
            f"""CREATE TABLE IF NOT EXISTS memory_vectors (
                   id         BIGSERIAL PRIMARY KEY,
                   collection TEXT NOT NULL,
                   text       TEXT NOT NULL,
                   metadata   JSONB NOT NULL DEFAULT '{{}}',
                   embedding  vector({settings.EMBED_DIM}),
                   tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
               )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS mv_collection_idx ON memory_vectors (collection)")
        c.execute("CREATE INDEX IF NOT EXISTS mv_tsv_idx ON memory_vectors USING GIN (tsv)")
        c.execute("ALTER TABLE memory_vectors ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now()")
        c.execute(
            """CREATE TABLE IF NOT EXISTS memory_profiles (
                   caller_id  TEXT PRIMARY KEY,
                   data       JSONB NOT NULL DEFAULT '{}',
                   updated_at timestamptz NOT NULL DEFAULT now()
               )"""
        )
        c.execute("ALTER TABLE memory_profiles ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()")
    return True


# ── erasure + retention (right-to-erasure / data-retention compliance) ───────
def pg_forget(caller_id: str, episodes_collection: str):
    """Delete everything tied to a caller: their (shared) profile + their episodes for
    the given collection + their contributions to global stores (by metadata.caller_id)."""
    _init_schema()
    with _conn() as c:
        c.execute("DELETE FROM memory_profiles WHERE caller_id=%s", (caller_id,))
        c.execute(
            "DELETE FROM memory_vectors WHERE collection=%s OR metadata->>'caller_id'=%s",
            (episodes_collection, caller_id),
        )


def pg_purge_older_than(days: int) -> int:
    _init_schema()
    with _conn() as c:
        n1 = c.execute(
            "DELETE FROM memory_vectors WHERE created_at < now() - make_interval(days => %s)", (days,)
        ).rowcount
        n2 = c.execute(
            "DELETE FROM memory_profiles WHERE updated_at < now() - make_interval(days => %s)", (days,)
        ).rowcount
    return n1 + n2


# ── profile / preferences / case (per-caller JSONB row) ──────────────────────
def pg_load_profile(caller_id: str) -> dict:
    _init_schema()
    with _conn() as c:
        row = c.execute("SELECT data FROM memory_profiles WHERE caller_id=%s", (caller_id,)).fetchone()
    return row[0] if row else {"profile": {}, "preferences": {}}


def pg_save_profile(caller_id: str, data: dict):
    _init_schema()
    with _conn() as c:
        c.execute(
            """INSERT INTO memory_profiles (caller_id, data) VALUES (%s, %s)
               ON CONFLICT (caller_id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()""",
            (caller_id, json.dumps(data)),
        )


# ── vectors (hybrid retrieval, same API as HybridStore) ──────────────────────
class PgHybridStore:
    def __init__(self, name: str):
        self.name = name
        _init_schema()

    @property
    def docs(self) -> list[dict]:
        """All rows for this collection (used by reflect())."""
        with _conn() as c:
            rows = c.execute(
                "SELECT text, metadata FROM memory_vectors WHERE collection=%s ORDER BY id", (self.name,)
            ).fetchall()
        return [{"text": t, "metadata": m} for t, m in rows]

    def add(self, text: str, metadata: dict | None = None):
        if not text or not text.strip():
            return
        emb = get_embeddings().embed_query(text)
        with _conn() as c:
            c.execute(
                "INSERT INTO memory_vectors (collection, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (self.name, text, json.dumps(metadata or {}), str(emb)),
            )

    def recall(self, query: str, k: int = 3, fetch_k: int = 20,
               hybrid: bool = True, rerank: bool = True) -> list[Document]:
        emb = str(get_embeddings().embed_query(query))
        with _conn() as c:
            dense = c.execute(
                """SELECT id, text, metadata FROM memory_vectors WHERE collection=%s
                   ORDER BY embedding <=> %s LIMIT %s""",
                (self.name, emb, fetch_k),
            ).fetchall()
            if not dense:
                return []
            rows = {r[0]: (r[1], r[2]) for r in dense}        # id -> (text, metadata)
            dense_ids = [r[0] for r in dense]

            if not hybrid:
                cand = dense_ids[:fetch_k]
                relevance = None
            else:
                sparse = c.execute(
                    """SELECT id, text, metadata FROM memory_vectors
                       WHERE collection=%s AND tsv @@ plainto_tsquery('english', %s)
                       ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC LIMIT %s""",
                    (self.name, query, query, fetch_k),
                ).fetchall()
                for r in sparse:
                    rows.setdefault(r[0], (r[1], r[2]))
                sparse_ids = [r[0] for r in sparse]
                fused = _rrf([dense_ids, sparse_ids])
                cand = sorted(fused, key=fused.get, reverse=True)[:fetch_k]
                relevance = fused

        if rerank and cand:
            scores = get_reranker().predict([(query, rows[i][0]) for i in cand])
            relevance = {i: float(s) for i, s in zip(cand, scores)}
            cand = [i for _, i in sorted(zip(scores, cand), key=lambda p: p[0], reverse=True)]

        out = []
        for i in cand[:k]:
            text, meta = rows[i]
            meta = dict(meta)
            meta["_relevance"] = float(relevance[i]) if relevance else 0.0
            out.append(Document(page_content=text, metadata=meta))
        return out
