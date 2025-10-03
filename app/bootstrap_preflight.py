# app/bootstrap_preflight.py
from __future__ import annotations

import os
import sys

def _open_collection(persist_dir: str, collection_name: str):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, coll

def rag_preflight() -> None:
    """
    Optional RAG self-check at startup.
    Enable by setting SELFTEST_RAG_QUERY. Other knobs:
      - CHROMA_PERSIST_DIR (or CHROMA_DB_DIR)
      - COLLECTION_NAME
      - SELFTEST_THRESHOLD (default 0.45)  # cosine distance
      - SELFTEST_MIN_COUNT (default 1)
      - SELFTEST_TOP_K (default 3)
    """
    q = os.getenv("SELFTEST_RAG_QUERY")
    if not q:
        return  # disabled

    persist = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    col_name = os.getenv("COLLECTION_NAME", "retie_docs")
    thr = float(os.getenv("SELFTEST_THRESHOLD", "0.45"))
    need = int(os.getenv("SELFTEST_MIN_COUNT", "1"))
    k = int(os.getenv("SELFTEST_TOP_K", "3"))

    # Use the same embedder used at indexing time
    from app.ingestion.embedder import embed_texts

    _, col = _open_collection(persist, col_name)

    try:
        q_emb = embed_texts([q])[0]
        res = col.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        dists = (res.get("distances") or [[]])[0]
        kept = [d for d in dists if d <= thr]
        count = 0
        try:
            count = col.count()
        except Exception:
            pass
        print(f"[SELFTEST] COUNT={count} dists={dists} kept<={thr}={len(kept)}")
        if len(kept) < need:
            print("[SELFTEST] FAILED: RAG no devuelve suficientes matches")
            # Optional hard-fail:
            # sys.exit(1)
    except Exception as e:
        print("[SELFTEST] ERROR:", e)
        # Optional hard-fail:
        # sys.exit(1)

def run() -> None:
    try:
        rag_preflight()
    except Exception as e:
        print("[SELFTEST] ERROR:", e)
        # Optional hard-fail:
        # sys.exit(1)
