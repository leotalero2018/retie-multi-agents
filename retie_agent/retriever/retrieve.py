# app/retriever/retrieve.py
from __future__ import annotations

import os
from typing import List, Dict, Optional

from retie_agent.retriever.chroma_client import get_collection
from retie_agent.llm.embedder import embed_texts

# Optional BM25 fallback
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None


def _env_top_k() -> int:
    try:
        return int(os.getenv("TOP_K", "5"))
    except Exception:
        return 5


def _env_distance_thr() -> float:
    try:
        return float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.75"))
    except Exception:
        return 0.75


def _bm25_fallback(query: str, documents: List[str], metas: List[Dict], top_k: int) -> List[Dict]:
    if not BM25Okapi or not documents:
        return []
    tokens_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokens_docs)
    scores = bm25.get_scores(query.split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "text": documents[i],
            "meta": metas[i],
            "score": float(scores[i]),
            "retrieval_method": "bm25",
            "score_type": "bm25_relevance",  # mayor = mejor
        }
        for i in idxs
    ]


def search(
    query: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    distance_threshold: Optional[float] = None,
) -> List[Dict]:
    """
    Dense-first retrieval:
      - Embeds the query with the SAME model used at index time.
      - Queries Chroma via query_embeddings (no EF bound).
      - Filters by a cosine distance threshold (lower is closer).
      - Falls back to BM25 if dense yields nothing (optional).
    Returns list of dicts: {"text": str, "meta": dict, "score": float(distance)}
    """
    k = top_k or _env_top_k()
    thr = _env_distance_thr() if distance_threshold is None else float(distance_threshold)

    col = get_collection(collection_name)

    # 1) Dense retrieval with explicit query embedding
    try:
        q_emb_vec = embed_texts([query])[0]  # must match dim/model used at index time
        res = col.query(
            query_embeddings=[q_emb_vec],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []

        dense_hits: List[Dict] = []
        for d, m, dist in zip(docs, metas, dists):
            if dist <= thr:
                dense_hits.append({
                    "text": d,
                    "meta": m,
                    "score": float(dist),
                    "retrieval_method": "dense",
                    "score_type": "cosine_distance",  # menor = mejor
                })

        if dense_hits:
            return dense_hits

        # if no good dense results, fall through to BM25
        raise RuntimeError("No dense hits under threshold.")

    except Exception:
        # 2) Optional BM25 lexical fallback
        try:
            res_all = col.get()  # returns {"ids":[], "documents":[], "metadatas":[]}
            documents: List[str] = res_all.get("documents", []) or []
            metas: List[Dict] = res_all.get("metadatas", []) or []
            return _bm25_fallback(query, documents, metas, k)
        except Exception:
            return []
