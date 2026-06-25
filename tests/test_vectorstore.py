"""HybridStore — dense + BM25 + rerank, persistence, guards (uses local embeddings)."""

from lang_memory.vectorstore import HybridStore


def test_add_and_recall():
    s = HybridStore("t")
    s.add("the cat sat on the mat", {"id": 1})
    s.add("dogs love to run in the park", {"id": 2})
    top = s.recall("a feline resting on a rug", k=1)
    assert top and top[0].metadata["id"] == 1


def test_empty_text_guard():
    s = HybridStore("t")
    s.add("")
    s.add("   ")
    assert s.docs == []


def test_recall_on_empty_store():
    assert HybridStore("t").recall("anything") == []


def test_persistence_reload():
    s = HybridStore("persist")
    s.add("a persistent fact about elephants", {"id": 9})
    s2 = HybridStore("persist")              # same DATA_DIR (test tmp) → reload from disk
    assert len(s2.docs) == 1
    assert s2.recall("elephant", k=1)[0].metadata["id"] == 9


def test_returned_metadata_clean():
    s = HybridStore("t")
    s.add("hello world", {"id": 1})
    doc = s.recall("hello", k=1)[0]
    assert "_i" not in doc.metadata          # internal id never leaks
    assert "_relevance" in doc.metadata       # score is exposed for blending


def test_dense_and_norerank_modes_work():
    s = HybridStore("t")
    for i in range(5):
        s.add(f"document number {i} with some content", {"id": i})
    assert s.recall("document", k=2, rerank=False)
    assert s.recall("document", k=2, hybrid=False)
