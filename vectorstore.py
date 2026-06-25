"""
vectorstore.py
──────────────
The retrieval engine shared by every memory store (episodic, failure patterns,
winning scripts, RAG knowledge base).

It is NOT a plain single-vector lookup. Pure dense search is weak on exact terms
(names, order ids, dates) — it thinks "order 4471" and "order 4472" are the same.
So this uses the retrieval stack that benchmarks best in this repo:

    dense (FAISS)  +  sparse (BM25)        ← two views of relevance
            └────────────┬────────────┘
                  RRF fusion             ← combine their rankings, no tuning
                       │
              cross-encoder rerank        ← reads (query, doc) pairs, fixes order
                       │
                    top-k

Same tiny API as before — `.add()` and `.recall()` — so the memory layers that
use it never had to change.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import settings, get_embeddings


@lru_cache(maxsize=1)
def get_reranker():
    """Shared cross-encoder — loaded once, reused by every store."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.RERANK_MODEL)


def _rrf(ranked_lists, k: int = 60) -> dict:
    """Reciprocal Rank Fusion: merge several ranked id-lists into one score map."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def make_store(name: str):
    """The configured store: Postgres if DATABASE_URL is set, else the file HybridStore.
    Same .add()/.recall()/.docs API either way, so the memory layers don't care which."""
    if settings.DATABASE_URL:
        from .pg_store import PgHybridStore   # lazy → psycopg only imported when used
        return PgHybridStore(name)
    return HybridStore(name)


class HybridStore:
    """One named, on-disk hybrid store. Created lazily on the first add()."""

    def __init__(self, name: str, base_dir: str | None = None):
        self.name = name
        self.dir = Path(base_dir or settings.DATA_DIR) / "hybrid" / name
        self.docs: list[dict] = []        # [{"text":.., "metadata":..}] — drives BM25 + rerank
        self._faiss: FAISS | None = None
        self._bm25: BM25Okapi | None = None
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    @property
    def _docs_path(self) -> Path:
        return self.dir / "docs.json"

    def _load(self):
        if (self.dir / "index.faiss").exists():
            self._faiss = FAISS.load_local(
                str(self.dir), get_embeddings(), allow_dangerous_deserialization=True)
        if self._docs_path.exists():
            self.docs = json.loads(self._docs_path.read_text(encoding="utf-8"))
            self._rebuild_bm25()

    def _rebuild_bm25(self):
        self._bm25 = BM25Okapi([d["text"].lower().split() for d in self.docs]) if self.docs else None

    # ── write ────────────────────────────────────────────────────────────────
    def add(self, text: str, metadata: dict | None = None):
        if not text or not text.strip():
            return
        meta = dict(metadata or {})
        meta["_i"] = len(self.docs)            # stable id so FAISS hits map back to self.docs
        self.docs.append({"text": text, "metadata": meta})

        if self._faiss is None:
            self._faiss = FAISS.from_texts([text], get_embeddings(), metadatas=[meta])
        else:
            self._faiss.add_texts([text], metadatas=[meta])
        self._rebuild_bm25()

        self.dir.mkdir(parents=True, exist_ok=True)
        self._faiss.save_local(str(self.dir))
        # Atomic write: a crash mid-write can never leave a corrupt docs.json
        # (which would otherwise crash on the next load).
        tmp = self._docs_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.docs, indent=2), encoding="utf-8")
        os.replace(tmp, self._docs_path)

    # ── read ─────────────────────────────────────────────────────────────────
    def recall(self, query: str, k: int = 3, fetch_k: int = 20,
               hybrid: bool = True, rerank: bool = True) -> list[Document]:
        """
        Return the k best documents. Each carries metadata["_relevance"] (the
        final score) so callers can blend it with recency/importance if they want.

          hybrid=False → dense-only (for benchmarking the upgrade)
          rerank=False → skip the cross-encoder (fusion order only)
        """
        if not self.docs or self._faiss is None:
            return []
        n = len(self.docs)
        fetch_k = min(fetch_k, n)

        dense = self._faiss.similarity_search(query, k=fetch_k)
        dense_idx = [d.metadata["_i"] for d in dense]

        if not hybrid:
            cand, relevance = dense_idx[:fetch_k], None
        else:
            bm = self._bm25.get_scores(query.lower().split())
            bm25_idx = sorted(range(n), key=lambda i: bm[i], reverse=True)[:fetch_k]
            fused = _rrf([dense_idx, bm25_idx])
            cand = sorted(fused, key=fused.get, reverse=True)[:fetch_k]
            relevance = fused

        if rerank and cand:
            pairs = [(query, self.docs[i]["text"]) for i in cand]
            scores = get_reranker().predict(pairs)
            relevance = {i: float(s) for i, s in zip(cand, scores)}
            cand = [i for _, i in sorted(zip(scores, cand), key=lambda p: p[0], reverse=True)]

        out = []
        for i in cand[:k]:
            meta = {kk: vv for kk, vv in self.docs[i]["metadata"].items() if kk != "_i"}
            meta["_relevance"] = float(relevance[i]) if relevance else 0.0
            out.append(Document(page_content=self.docs[i]["text"], metadata=meta))
        return out
