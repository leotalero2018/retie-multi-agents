# app/retriever/retrieve.py
from typing import List, Dict, Optional
from app.retriever.chroma_client import get_collection
from app.ingestion.embedder import embed_texts
from app.config import settings

# Fallback BM25 (opcional)
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


def _bm25_fallback(query: str, documents: List[str], metas: List[Dict], top_k: int) -> List[Dict]:
    if not BM25Okapi:
        return []
    tokens_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokens_docs)
    scores = bm25.get_scores(query.split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"text": documents[i], "meta": metas[i], "score": float(scores[i])} for i in idxs]


def search(
    query: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    distance_threshold: Optional[float] = None,
) -> List[Dict]:
    """
    Dense-first retrieval:
    - Embeds the query with the SAME model used at index time.
    - Queries Chroma via query_embeddings (no EF conflict).
    - Applies a distance threshold to filter weak hits.
    - Falls back to BM25 if dense gives nothing (optional).
    """
    top_k = top_k or settings.TOP_K
    thr = distance_threshold if distance_threshold is not None else getattr(settings, "RAG_DISTANCE_THRESHOLD", 0.45)

    col = get_collection(collection_name)  # IMPORTANT: must NOT attach embedding_function here

    # 1) Dense RAG
    try:
        # If you indexed with OpenAI 1536-dim, DO NOT prefix "query:" to the text.
        q_emb_vec = embed_texts([query])[0]  # must return a 1536-dim vector (same model as index)
        res = col.query(
            query_embeddings=[q_emb_vec],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []

        items_dense = []
        for d, m, dist in zip(docs, metas, dists):
            if dist <= thr:
                items_dense.append({"text": d, "meta": m, "score": float(dist)})

        # Logging útil en depuración:
        # print(f"[RAG] k={top_k} thr={thr} dists={dists} kept={len(items_dense)}")

        if items_dense:
            return items_dense

        # si no hay resultados densos útiles, intenta fallback
        raise RuntimeError("empty or weak dense results")

    except Exception:
        # 2) Fallback léxico BM25 si está disponible
        try:
            res_all = col.get()  # dict plano con 'documents' y 'metadatas'
            documents: List[str] = res_all.get("documents", []) or []
            metas: List[Dict] = res_all.get("metadatas", []) or []
            if BM25Okapi and documents:
                return _bm25_fallback(query, documents, metas, top_k)
        except Exception:
            pass
        return []
