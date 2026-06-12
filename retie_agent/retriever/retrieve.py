# app/retriever/retrieve.py
from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from retie_agent.config import settings, as_bool
from retie_agent.retriever.chroma_client import get_collection, list_collection_names
from retie_agent.llm.embedder import embed_texts

logger = logging.getLogger(__name__)

# Optional BM25 (lexical) support
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None


# ──────────────────────────────────────────────────────────────────────────────
# Query embedding cache (LRU) — evita re-embeber la misma consulta
# (p. ej. /docs tras una pregunta, o reintentos) y ahorra latencia/costo.
# ──────────────────────────────────────────────────────────────────────────────
_EMB_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_EMB_CACHE_MAX = 256
_EMB_LOCK = threading.Lock()


def _embed_query_cached(query: str) -> List[float]:
    key = query.strip().lower()
    with _EMB_LOCK:
        if key in _EMB_CACHE:
            _EMB_CACHE.move_to_end(key)
            return _EMB_CACHE[key]
    vec = embed_texts([query])[0]
    with _EMB_LOCK:
        _EMB_CACHE[key] = vec
        _EMB_CACHE.move_to_end(key)
        while len(_EMB_CACHE) > _EMB_CACHE_MAX:
            _EMB_CACHE.popitem(last=False)
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# BM25 index cache — antes se llamaba col.get() (colección COMPLETA) en cada
# fallback; ahora el corpus se carga una vez por colección y se reconstruye
# solo si cambia el número de documentos.
# ──────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class _BM25Index:
    def __init__(self, documents: List[str], metas: List[Dict], count: int) -> None:
        self.documents = documents
        self.metas = metas
        self.count = count
        self.bm25 = BM25Okapi([_tokenize(d) for d in documents]) if (BM25Okapi and documents) else None
        self.built_at = time.time()


_BM25_CACHE: Dict[str, _BM25Index] = {}
_BM25_LOCK = threading.Lock()


def _get_bm25_index(col, col_name: str) -> Optional[_BM25Index]:
    if BM25Okapi is None:
        return None
    try:
        count = col.count()
    except Exception:
        count = -1
    cached = _BM25_CACHE.get(col_name)
    if cached is not None and cached.count == count:
        return cached
    with _BM25_LOCK:
        cached = _BM25_CACHE.get(col_name)
        if cached is not None and cached.count == count:
            return cached
        try:
            res_all = col.get(include=["documents", "metadatas"])
            documents: List[str] = res_all.get("documents", []) or []
            metas: List[Dict] = res_all.get("metadatas", []) or []
            idx = _BM25Index(documents, metas, count)
            _BM25_CACHE[col_name] = idx
            logger.info("BM25 index built for '%s' (%d docs)", col_name, len(documents))
            return idx
        except Exception as exc:
            logger.warning("BM25 index build failed for '%s': %s", col_name, exc)
            return None


def _bm25_search(idx: _BM25Index, query: str, top_k: int) -> List[Dict]:
    if not idx or not idx.bm25:
        return []
    scores = idx.bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "text": idx.documents[i],
            "meta": idx.metas[i] or {},
            "score": float(scores[i]),
            "retrieval_method": "bm25",
            "score_type": "bm25_relevance",  # mayor = mejor
        }
        for i in order
        if scores[i] > 0
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Referencias estructurales: "artículo 20.3", "tabla 13.1", "anexo general"…
# Cuando el usuario nombra una referencia exacta, una búsqueda literal con
# where_document es mucho más precisa que la semántica.
# ──────────────────────────────────────────────────────────────────────────────
_STRUCT_REF_RE = re.compile(
    r"\b(art[ií]culo|tabla|secci[oó]n|cap[ií]tulo|anexo)\s+(\d+(?:[.\-]\d+)*|[ivxlcdm]+\b|general\b)",
    re.IGNORECASE,
)

_STRUCT_CANON = {
    "articulo": "ARTÍCULO",
    "artículo": "ARTÍCULO",
    "tabla": "TABLA",
    "seccion": "SECCIÓN",
    "sección": "SECCIÓN",
    "capitulo": "CAPÍTULO",
    "capítulo": "CAPÍTULO",
    "anexo": "ANEXO",
}


def _struct_refs(query: str) -> List[Tuple[str, str]]:
    """Devuelve [(tipo_canónico, número)] de las referencias citadas en la consulta."""
    refs: List[Tuple[str, str]] = []
    for m in _STRUCT_REF_RE.finditer(query or ""):
        kind = _STRUCT_CANON.get(m.group(1).lower())
        if kind:
            refs.append((kind, m.group(2).upper() if m.group(2).isalpha() else m.group(2)))
    return refs[:3]  # límite defensivo


def _struct_variants(kind: str, num: str) -> List[str]:
    """Variantes de mayúsculas/acentos con las que el texto puede aparecer indexado."""
    title = kind.capitalize()  # "Artículo"
    plain = kind.replace("Í", "I").replace("Ó", "O")  # sin tilde, mayúsculas
    variants = {f"{kind} {num}", f"{title} {num}", f"{plain} {num}", f"{plain.capitalize()} {num}"}
    return list(variants)


def _struct_search(col, query_emb: List[float], refs: List[Tuple[str, str]], top_k: int) -> List[Dict]:
    """Búsqueda semántica restringida a chunks que contienen la referencia literal."""
    hits: List[Dict] = []
    for kind, num in refs:
        contains = [{"$contains": v} for v in _struct_variants(kind, num)]
        where_doc: Dict[str, Any] = contains[0] if len(contains) == 1 else {"$or": contains}
        try:
            res = col.query(
                query_embeddings=[query_emb],
                n_results=max(3, top_k),
                where_document=where_doc,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:
            logger.debug("struct search failed for %s %s: %s", kind, num, exc)
            continue
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []
        for d, m, dist in zip(docs, metas, dists):
            hits.append({
                "text": d,
                "meta": m or {},
                "score": float(dist),
                "retrieval_method": "struct",
                "score_type": "cosine_distance",
            })
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Fusión RRF (Reciprocal Rank Fusion) de listas dense / bm25 / struct.
# ──────────────────────────────────────────────────────────────────────────────
def _hit_key(h: Dict) -> Tuple:
    meta = h.get("meta", {}) or {}
    cid = meta.get("chunk_id")
    if cid is not None:
        return (meta.get("source"), meta.get("page"), cid)
    return (meta.get("source"), meta.get("page"), hash(h.get("text", "")))


def _rrf_fuse(ranked_lists: List[Tuple[List[Dict], float]], top_k: int, rrf_k: int) -> List[Dict]:
    """Combina varias listas rankeadas. Cada hit conserva su score/score_type
    nativos; el orden final lo decide el puntaje RRF acumulado."""
    fused: Dict[Tuple, Dict] = {}
    scores: Dict[Tuple, float] = {}
    for hits, weight in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            key = _hit_key(h)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            # Preferir la versión "dense"/"struct" del hit (trae distancia coseno).
            if key not in fused or h.get("score_type") == "cosine_distance":
                fused[key] = h
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict] = []
    for key, rrf_score in ordered[:top_k]:
        hit = dict(fused[key])
        hit["rrf_score"] = round(rrf_score, 5)
        out.append(hit)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Normalización de metadata: el pipeline de normativas indexa doc_name/pagina,
# mientras el agente (dedupe, citas, trazas) espera source/page. Se completan
# las claves esperadas sin perder las originales.
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_meta(meta: Dict) -> Dict:
    if not isinstance(meta, dict):
        return {}
    if not meta.get("source"):
        src = meta.get("doc_name") or meta.get("doc_id")
        if src:
            meta["source"] = src
    if meta.get("page") is None and meta.get("pagina") is not None:
        meta["page"] = meta["pagina"]
    return meta


def _finalize_hits(hits: List[Dict]) -> List[Dict]:
    for h in hits:
        h["meta"] = _normalize_meta(h.get("meta", {}) or {})
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Resolución de colecciones: acepta nombres separados por coma y, si ninguna
# de las configuradas existe (p. ej. COLLECTION_NAME obsoleto tras reindexar),
# cae a las colecciones realmente presentes en la DB en vez de devolver vacío.
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_collections(collection_name: Optional[str]) -> List[Tuple[str, Any]]:
    requested = collection_name or getattr(settings, "COLLECTION_NAMES", "") or settings.COLLECTION_NAME
    names = [n.strip() for n in str(requested).split(",") if n.strip()]

    cols: List[Tuple[str, Any]] = []
    for name in names:
        try:
            cols.append((name, get_collection(name)))
        except Exception as exc:
            logger.warning("Collection '%s' no disponible: %s", name, exc)

    if not cols:
        available = list_collection_names()
        if available:
            logger.warning(
                "Ninguna de las colecciones configuradas %s existe; usando las disponibles: %s",
                names, available,
            )
            for name in available:
                try:
                    cols.append((name, get_collection(name)))
                except Exception:
                    pass
    return cols


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────
def search(
    query: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    distance_threshold: Optional[float] = None,
) -> List[Dict]:
    """Recuperación híbrida sobre una o varias colecciones:
      1. Dense (embedding de la consulta, mismo modelo del índice) con umbral
         de distancia coseno (settings.RAG_DISTANCE_THRESHOLD).
      2. BM25 léxico sobre un índice cacheado por colección.
      3. Búsqueda dirigida cuando la consulta cita "artículo/tabla/anexo N".
      `collection_name` admite varios nombres separados por coma; las listas
      se combinan con Reciprocal Rank Fusion. Devuelve a lo sumo top_k hits.
    """
    k = top_k or settings.TOP_K
    thr = float(distance_threshold) if distance_threshold is not None else settings.RAG_DISTANCE_THRESHOLD
    fetch_k = max(k, k * max(1, getattr(settings, "FETCH_K_MULTIPLIER", 3)))
    fusion_enabled = as_bool(getattr(settings, "BM25_FUSION_ENABLED", "true"))
    rrf_k = int(getattr(settings, "RRF_K", 60))

    cols = _resolve_collections(collection_name)
    if not cols:
        logger.error("No hay colecciones disponibles en la DB de Chroma")
        return []

    query_emb: Optional[List[float]] = None
    try:
        query_emb = _embed_query_cached(query)
    except Exception as exc:
        logger.warning("Query embedding failed (%s); relying on BM25 only", exc)

    # 1) Dense retrieval (las distancias coseno son comparables entre colecciones
    #    porque comparten el modelo de embeddings → merge directo por score)
    dense_hits: List[Dict] = []
    if query_emb is not None:
        for col_name, col in cols:
            try:
                res = col.query(
                    query_embeddings=[query_emb],
                    n_results=fetch_k,
                    include=["documents", "distances", "metadatas"],
                )
                docs = res.get("documents", [[]])[0] or []
                metas = res.get("metadatas", [[]])[0] or []
                dists = res.get("distances", [[]])[0] or []
                for d, m, dist in zip(docs, metas, dists):
                    if dist <= thr:
                        dense_hits.append({
                            "text": d,
                            "meta": m or {},
                            "score": float(dist),
                            "retrieval_method": "dense",
                            "score_type": "cosine_distance",  # menor = mejor
                            "collection": col_name,
                        })
            except Exception as exc:
                logger.warning("Dense retrieval failed on '%s': %s", col_name, exc)
        dense_hits.sort(key=lambda h: h["score"])
        dense_hits = dense_hits[:fetch_k]

    # 2) BM25 léxico (índice cacheado por colección; cada colección aporta su
    #    propia lista rankeada — los scores BM25 no son comparables entre corpus)
    bm25_lists: List[List[Dict]] = []
    if BM25Okapi is not None and (fusion_enabled or not dense_hits):
        for col_name, col in cols:
            idx = _get_bm25_index(col, col_name)
            if idx is not None:
                hits = _bm25_search(idx, query, fetch_k)
                if hits:
                    for h in hits:
                        h["collection"] = col_name
                    bm25_lists.append(hits)

    # 3) Búsqueda dirigida por referencia estructural ("artículo 20.3", "tabla 13.1")
    struct_hits: List[Dict] = []
    refs = _struct_refs(query)
    if refs and query_emb is not None:
        for col_name, col in cols:
            col_struct = _struct_search(col, query_emb, refs, k)
            for h in col_struct:
                h["collection"] = col_name
            struct_hits.extend(col_struct)
        struct_hits.sort(key=lambda h: h["score"])

    # Fusión / fallback
    lists: List[Tuple[List[Dict], float]] = []
    if dense_hits:
        lists.append((dense_hits, 1.0))
    for bl in bm25_lists:
        lists.append((bl, 0.7))
    if struct_hits:
        lists.append((struct_hits, 1.2))  # la cita literal pesa más

    if not lists:
        return []
    if len(lists) == 1:
        return _finalize_hits(lists[0][0][:k])
    return _finalize_hits(_rrf_fuse(lists, k, rrf_k))
