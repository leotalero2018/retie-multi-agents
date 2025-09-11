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


def search(query: str, top_k: Optional[int] = None, collection_name: Optional[str] = None) -> List[Dict]:
    top_k = top_k or settings.TOP_K
    col = get_collection(collection_name)

    # 1) RAG denso (embeddings)
    try:
        # Usa una sola forma. Si indexaste con OpenAI embeddings, no uses prefijo "query: "
        q_emb = embed_texts([query])[0]
        res = col.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        items = [{"text": d, "meta": m, "score": float(dist)} for d, m, dist in zip(docs, metas, dists)]
        if items:
            return items
        # si no hay resultados, intenta fallback
        raise RuntimeError("empty dense results")
    except Exception:
        # 2) Fallback léxico BM25 si hay paquete y la colección no es enorme
        try:
            res_all = col.get()   # dict con 'documents' y 'metadatas' planas (listas del mismo largo)
            documents: List[str] = res_all.get("documents", []) or []
            metas: List[Dict] = res_all.get("metadatas", []) or []
            if BM25Okapi and documents:
                return _bm25_fallback(query, documents, metas, top_k)
        except Exception:
            pass
        return []
