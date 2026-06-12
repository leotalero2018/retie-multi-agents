# app/agent/graph.py
from __future__ import annotations
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from openai import OpenAI

from retie_agent.config import settings, as_bool
from retie_agent.retriever.retrieve import search
from retie_agent.agent.prompt import (
    make_prompt,
    make_hybrid_prompt,
    SYSTEM_PROMPT_RETIE,
    CONDENSE_PROMPT,
    format_history_for_condense,
)
from retie_agent.agent.retie_agent import _dedupe_hits, _resolve_collection, _resolve_model
from retie_agent.observability.obs import trace_ctx, span_ctx, log_generation
from retie_agent.services.history import get_history, add_message
from retie_agent.agent.notebooklm_client import NotebookLMClient, NotebookLMError, NotebookLMCache
from retie_agent.agent.table_render import render_telegram_table, render_table_image

# Executor compartido para trabajo paralelo (Chroma + NotebookLM, enrichment con
# timeout). Es persistente a propósito: un `with ThreadPoolExecutor(...)` espera
# en el __exit__ a los futures en ejecución, lo que anulaba el "skip" de
# NotebookLM (la respuesta quedaba bloqueada hasta que el navegador terminara).
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="graph")

# ---------- State ----------
class GraphState(TypedDict, total=False):
    question: str
    user_id: str
    session: str
    agent_key: Optional[str]
    hits: List[Dict[str, Any]]
    route: str
    answer: Any   # may now hold dict for styled payload
    metadata: Optional[Dict[str, Any]]  # new, for passing channel info
    history: List[Dict[str, str]]
    wants_table: bool       # user asked for a "tabla" → route through table_node
    table_image: Any        # PNG bytes when the table is rendered as an image
    nlm_answer: Optional[str]  # NotebookLM answer from parallel hybrid retrieve
    search_query: Optional[str]  # standalone query (condensed from history) used for retrieval


# Detección de intención de tabla por palabra clave (determinística).
_TABLE_RE = re.compile(r"\btablas?\b", re.IGNORECASE)


def _wants_table(text: str) -> bool:
    return bool(_TABLE_RE.search(text or ""))


# Saludos / cortesías: se responden al instante, sin retrieval ni NotebookLM.
# (Antes "Hola" disparaba el pipeline completo y esperaba al navegador de NLM
# para terminar en "No tengo evidencia en los documentos".)
_SMALLTALK_RE = re.compile(
    r"^\s*(?:hola+|holi+|buen[oa]s(?:\s+(?:d[ií]as|tardes|noches))?|hey|hello|hi"
    r"|(?:muchas\s+)?gracias+|ok(?:ey)?|vale|listo|perfecto|genial|excelente"
    r"|adi[oó]s|hasta\s+luego|chao|nos\s+vemos"
    r"|qu[ié][eé]n\s+eres|qu[eé]\s+puedes\s+hacer|ayuda)\s*[!.?¡¿]*\s*$",
    re.IGNORECASE,
)

_SMALLTALK_THANKS_RE = re.compile(
    r"gracias|adi[oó]s|hasta\s+luego|chao|nos\s+vemos|ok|vale|listo|perfecto|genial|excelente",
    re.IGNORECASE,
)

_SMALLTALK_GREETING = (
    "¡Hola! 👋 Soy el asistente de normativa eléctrica colombiana (RETIE y NTC 2050).\n\n"
    "Pregúntame, por ejemplo:\n"
    "• ¿Qué exige el RETIE sobre puesta a tierra?\n"
    "• Dame la tabla 220.55 de factores de demanda\n"
    "• Requisitos para instalaciones en zonas húmedas\n\n"
    "Escribe /help para ver todos los comandos."
)
_SMALLTALK_THANKS = "¡Con gusto! 🙌 Si tienes otra consulta sobre el RETIE o la NTC 2050, aquí estoy."


def _node_smalltalk(state: GraphState) -> GraphState:
    q = state.get("question", "")
    answer = _SMALLTALK_THANKS if _SMALLTALK_THANKS_RE.search(q) else _SMALLTALK_GREETING
    with span_ctx(None, "smalltalk_node", as_type="chain", span_input={"question": q}):
        return {"answer": answer}


def make_state(question: str, *, user_id: str = "anon", session: str = "default", agent_key: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> GraphState:
    return {
        "question": (question or "").strip(),
        "user_id": user_id or "anon",
        "session": session or "default",
        "agent_key": agent_key,
        "history": history or [],
    }

_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))

# ---------- Trace helpers ----------
def _format_chunks_for_trace(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convierte los hits del retriever en documentos estructurados y legibles
    para la vista de Langfuse: rank, score, fuente, página, chunk_id y contenido."""
    docs: List[Dict[str, Any]] = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {}) or {}
        score = h.get("score")
        docs.append({
            "rank": i,
            "score": round(score, 4) if isinstance(score, (int, float)) else score,
            "score_type": h.get("score_type", "cosine_distance"),
            "retrieval_method": h.get("retrieval_method", "dense"),
            "source": meta.get("source", "desconocido"),
            "page": meta.get("page", meta.get("page_number")),
            "chunk_id": meta.get("chunk_id"),
            "content": h.get("text", ""),
        })
    return docs


def _retrieval_stats(hits: List[Dict[str, Any]], coll: str, top_k: int, thr: float) -> Dict[str, Any]:
    scores = [h.get("score") for h in hits if isinstance(h.get("score"), (int, float))]
    method = hits[0].get("retrieval_method", "dense") if hits else "none"
    sources = []
    for h in hits:
        s = (h.get("meta", {}) or {}).get("source")
        if s and s not in sources:
            sources.append(s)
    stats: Dict[str, Any] = {
        "collection": coll,
        "top_k": top_k,
        "distance_threshold": thr,
        "retrieval_method": method,
        "hits_count": len(hits),
        "unique_sources": sources,
    }
    if scores:
        stats["score_best"] = round(min(scores), 4) if method == "dense" else round(max(scores), 4)
        stats["score_worst"] = round(max(scores), 4) if method == "dense" else round(min(scores), 4)
    return stats


# ---------- Citas / Fuentes para el usuario ----------
# Marcadores estructurales del RETIE para extraer la sección de forma best-effort
# (la metadata indexada solo tiene source/page/chunk_id; la sección se infiere del texto).
_SECTION_PATTERNS = [
    r"ART[IÍ]CULO\s+\d+[°ºo]?",
    r"SECCI[OÓ]N\s+\d+(?:\.\d+)*",
    r"CAP[IÍ]TULO\s+[IVXLCDM0-9]+",
    r"T[IÍ]TULO\s+[IVXLCDM0-9]+",
    r"TABLA\s+\d+(?:[.\-]\d+)*",
    r"ANEXO\s+[A-Z0-9]+",
]
_SECTION_RE = [re.compile(p, re.IGNORECASE) for p in _SECTION_PATTERNS]


def _extract_section(text: str) -> Optional[str]:
    """Intenta inferir la sección/artículo del RETIE a partir del texto del chunk.
    Retorna None si no encuentra un marcador estructural reconocible."""
    if not text:
        return None
    for rx in _SECTION_RE:
        m = rx.search(text)
        if m:
            return " ".join(m.group(0).split())
    return None


def _clean_source_name(name: Any) -> str:
    s = str(name or "Documento")
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    return s.strip()


def _build_user_sources(hits: List[Dict[str, Any]], limit: int = 5):
    """Construye las citas para el usuario a partir de los hits reales del retriever.
    Deduplica por (fuente, página), infiere la sección y devuelve:
      - sources: lista estructurada [{source, page, section}]
      - sources_text: bloque de texto listo para mostrar al usuario.
    """
    seen = set()
    items: List[Dict[str, Any]] = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        source = _clean_source_name(meta.get("source"))
        page = meta.get("page", meta.get("page_number"))
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "source": source,
            "page": page,
            "section": _extract_section(h.get("text", "")),
        })
        if len(items) >= limit:
            break

    if not items:
        return [], ""

    lines = ["📚 Fuentes consultadas:"]
    for i, it in enumerate(items, start=1):
        parts = [it["source"]]
        if it["page"] is not None:
            parts.append(f"pág. {it['page']}")
        if it["section"]:
            parts.append(it["section"])
        lines.append(f"{i}. " + " · ".join(parts))
    return items, "\n".join(lines)


# ---------- Nodes ----------
def _node_condense(state: GraphState) -> GraphState:
    """Reescribe preguntas de seguimiento como preguntas autocontenidas.

    "¿Y eso aplica en baja tensión?" no recupera nada útil en Chroma/NotebookLM
    sin el contexto previo. Con historial, una llamada corta al LLM produce la
    consulta de búsqueda; la pregunta original se conserva para el prompt final.
    Sin historial (o con el flag apagado) es un pass-through sin costo.
    """
    q = state["question"]
    history = state.get("history") or []
    enabled = as_bool(getattr(settings, "QUERY_REWRITE_ENABLED", "true"))
    if not enabled or not history:
        return {"search_query": q}

    with span_ctx(
        None, "condense_node", as_type="chain",
        span_input={"question": q, "history_messages": len(history)},
    ) as span:
        search_query = q
        try:
            prompt = CONDENSE_PROMPT.format(
                history=format_history_for_condense(history), question=q
            )
            resp = _client.chat.completions.create(
                model=_resolve_model(state.get("agent_key"), explicit=None),
                temperature=0.0,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = (resp.choices[0].message.content or "").strip().strip('"')
            # Sanidad: descarta reescrituras vacías o desproporcionadas.
            if rewritten and len(rewritten) <= max(300, len(q) * 4):
                search_query = rewritten
        except Exception as exc:
            logger.warning("condense_node failed, using original question: %s", exc)

        if span is not None:
            try:
                span.update(output={
                    "search_query": search_query,
                    "rewritten": search_query != q,
                })
            except Exception:
                pass
        return {"search_query": search_query}


def _node_retrieve(state: GraphState) -> GraphState:
    q = state.get("search_query") or state["question"]
    agent_key = state.get("agent_key")
    coll = _resolve_collection(agent_key, explicit=None)
    # Tablas: más chunks — las tablas largas viven partidas en varios fragmentos.
    top_k = (
        getattr(settings, "TOP_K_TABLES", 12)
        if state.get("wants_table")
        else getattr(settings, "TOP_K", 4)
    )
    thr = getattr(settings, "RAG_DISTANCE_THRESHOLD", 0.45)

    span_input = {
        "question": q,
        "collection": coll,
        "top_k": top_k,
        "distance_threshold": thr,
    }
    with span_ctx(None, "retrieve", as_type="retriever", span_input=span_input) as span:
        try:
            hits = _dedupe_hits(search(q, top_k=top_k, collection_name=coll))
        except Exception:
            hits = []

        if span is not None:
            try:
                stats = _retrieval_stats(hits, coll, top_k, thr)
                # output = documentos recuperados (Langfuse los renderiza como lista)
                span.update(
                    output={"documents": _format_chunks_for_trace(hits)},
                    metadata=stats,
                )
            except Exception:
                pass
        return {"hits": hits}

def _node_route_entry(state: GraphState) -> GraphState:
    """Entry point: smalltalk directo, o hybrid (Chroma+NLM) / Chroma-only según flag."""
    q = state.get("question", "")
    nlm_enabled = str(getattr(settings, "NOTEBOOKLM_ENABLED", "false")).lower() in ("1", "true", "yes")
    if _SMALLTALK_RE.match(q):
        route = "smalltalk"
    else:
        # hybrid_retrieve runs Chroma + NLM in parallel with caching; falls back to Chroma-only
        route = "hybrid_retrieve" if nlm_enabled else "retrieve"
    wants_table = _wants_table(q)
    with span_ctx(
        None, "route_entry", as_type="chain",
        span_input={"nlm_enabled": nlm_enabled, "wants_table": wants_table},
    ) as span:
        if span is not None:
            try:
                span.update(output={"route": route, "wants_table": wants_table})
            except Exception:
                pass
    return {"route": route, "wants_table": wants_table}


def _node_router(state: GraphState) -> GraphState:
    hits = state.get("hits") or []
    route = "answer_node" if hits else "no_context"
    with span_ctx(
        None, "router", as_type="chain",
        span_input={"hits_count": len(hits)},
    ) as span:
        if span is not None:
            try:
                span.update(output={"route": route})
            except Exception:
                pass
        return {"route": route}

def _node_answer(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")
    hits = state.get("hits") or []
    nlm_answer = (state.get("nlm_answer") or "").strip()

    if not hits and not nlm_answer:
        return {"answer": "No tengo evidencia en los documentos."}

    model = _resolve_model(agent_key, explicit=None)
    sys = SYSTEM_PROMPT_RETIE
    # Use hybrid prompt when both sources are available
    prompt = make_hybrid_prompt(hits, nlm_answer, q, is_admin=False)
    messages = [{"role": "system", "content": sys}]
    for msg in state.get("history", []):
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})

    temperature = 0.0
    # Tables need more tokens to reproduce ALL rows; regular answers stay at default.
    wants_table = state.get("wants_table", False)
    default_max = getattr(settings, "MAX_TOKENS", 600)
    max_tokens = 3000 if wants_table else default_max

    with span_ctx(
        None, "answer_node",
        {"model": model, "context_chunks": len(hits), "agent_key": agent_key or "default"},
        as_type="chain",
        span_input={"question": q},
    ) as span:
        resp = _client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        answer = (resp.choices[0].message.content or "").strip()

        try:
            usage = getattr(resp, "usage", None)
            usage_dict = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
            log_generation(
                None,
                name="openai.chat",
                input_text=messages,  # lista de mensajes → Langfuse la renderiza como chat
                output_text=answer,
                model=model,
                usage=usage_dict,
                metadata={
                    "agent_key": agent_key or "default",
                    "context_chunks": len(hits),
                    "sources": [(h.get("meta", {}) or {}).get("source") for h in hits],
                },
                model_parameters={"temperature": temperature, "max_tokens": max_tokens},
            )
        except Exception:
            pass

        if span is not None:
            try:
                span.update(output={"answer": answer})
            except Exception:
                pass

        return {"answer": answer}

def _node_no_context(_state: GraphState) -> GraphState:
    with span_ctx(None, "no_context", as_type="chain"):
        return {"answer": "No tengo evidencia en los documentos."}


# ===============================
#  NotebookLM MCP Node (legacy solo-NLM path, kept for reference)
# ===============================
_notebooklm_client: Optional[NotebookLMClient] = None


def _get_notebooklm_client() -> NotebookLMClient:
    global _notebooklm_client
    if _notebooklm_client is None:
        _notebooklm_client = NotebookLMClient(
            base_url=getattr(settings, "NOTEBOOKLM_URL", "http://localhost:3000"),
            notebook_id=getattr(settings, "NOTEBOOKLM_NOTEBOOK_ID", None),
            timeout=float(getattr(settings, "NOTEBOOKLM_TIMEOUT", 120.0)),
            page_timeout_ms=int(getattr(settings, "NOTEBOOKLM_PAGE_TIMEOUT_MS", 15000)),
        )
    return _notebooklm_client


# ===============================
#  Hybrid RAG Node (Chroma + NLM in parallel with cache)
# ===============================
_nlm_cache: Optional[NotebookLMCache] = None


def _get_nlm_cache() -> NotebookLMCache:
    global _nlm_cache
    if _nlm_cache is None:
        ttl = int(getattr(settings, "NOTEBOOKLM_CACHE_TTL", 3600))
        _nlm_cache = NotebookLMCache(ttl=ttl)
    return _nlm_cache


def _node_hybrid_retrieve(state: GraphState) -> GraphState:
    """Hybrid RAG: runs Chroma and NotebookLM in parallel, merges both results.

    Fast-path exits:
    - Cache hit on NLM → answer available immediately, no browser wait.
    - Chroma high-confidence (best score < CHROMA_HIGH_CONFIDENCE_THR) → NLM skipped.

    Otherwise waits up to NOTEBOOKLM_PARALLEL_TIMEOUT seconds for NLM, then
    falls back to Chroma-only. Always routes to answer_node (merged LLM call)
    or no_context when both sources are empty.
    """
    q = state.get("search_query") or state["question"]
    agent_key = state.get("agent_key")
    coll = _resolve_collection(agent_key, explicit=None)
    # Tablas: más chunks — las tablas largas viven partidas en varios fragmentos.
    top_k = (
        getattr(settings, "TOP_K_TABLES", 12)
        if state.get("wants_table")
        else getattr(settings, "TOP_K", 4)
    )
    nb_id = getattr(settings, "NOTEBOOKLM_NOTEBOOK_ID", None)
    conf_thr = float(getattr(settings, "CHROMA_HIGH_CONFIDENCE_THR", 0.25))
    decent_thr = float(getattr(settings, "CHROMA_DECENT_THR", 0.40))
    parallel_timeout = float(getattr(settings, "NOTEBOOKLM_PARALLEL_TIMEOUT", 25.0))
    soft_timeout = float(getattr(settings, "NOTEBOOKLM_SOFT_TIMEOUT", 12.0))
    min_chars = int(getattr(settings, "NOTEBOOKLM_MIN_QUERY_CHARS", 12))
    sem_sim = float(getattr(settings, "NLM_SEMANTIC_CACHE_SIM", 0.93))
    # Modo "siempre ambos": no se salta NLM ni se recorta su espera.
    always_wait = as_bool(getattr(settings, "NOTEBOOKLM_ALWAYS_WAIT", "true"))
    hard_timeout = float(getattr(settings, "NOTEBOOKLM_HARD_TIMEOUT", 180.0))

    nlm_eligible = len(q.strip()) >= min_chars
    cache = _get_nlm_cache()
    cache_kind = None
    cached = cache.get(q, nb_id) if nlm_eligible else None
    if cached:
        cache_kind = "exact"

    # Caché semántico: una pregunta reformulada pero equivalente reutiliza la
    # respuesta NLM previa. El embedding se comparte vía LRU con el retrieval
    # de Chroma (no hay llamada extra a la API en el camino feliz).
    q_emb: Optional[List[float]] = None
    if nlm_eligible and not cached and sem_sim > 0:
        try:
            from retie_agent.retriever.retrieve import _embed_query_cached
            q_emb = _embed_query_cached(q)
            cached = cache.get_semantic(q_emb, nb_id, sem_sim)
            if cached:
                cache_kind = "semantic"
        except Exception:
            q_emb = None

    with span_ctx(
        None, "hybrid_retrieve", as_type="retriever",
        span_input={"question": q, "nlm_cache_hit": cache_kind},
    ) as span:
        hits: List[Dict[str, Any]] = []
        nlm_answer = ""
        nlm_status = "disabled"

        # Pool persistente: si se decide saltar/abandonar NotebookLM, el future
        # queda corriendo en background sin bloquear esta respuesta.
        chroma_fut = _POOL.submit(
            lambda: _dedupe_hits(search(q, top_k=top_k, collection_name=coll))
        )

        nlm_fut = None
        if nlm_eligible and not cached:
            client = _get_notebooklm_client()
            nlm_fut = _POOL.submit(client.ask_question, q, "footnotes", nb_id)
            nlm_status = "submitted"

        # Chroma is fast — retrieve results first
        try:
            hits = chroma_fut.result(timeout=15)
        except Exception:
            hits = []

        # Confidence check: skip NLM wait if Chroma is highly confident.
        # EXCEPTION: never skip NLM for table queries — table data requires
        # NotebookLM's full-PDF access; Chroma chunks rarely contain full tables.
        wants_table = state.get("wants_table", False)
        # Solo distancias coseno: los scores BM25 (mayor=mejor) no son comparables.
        scores = [
            h["score"] for h in hits
            if isinstance(h.get("score"), (int, float))
            and h.get("score_type", "cosine_distance") == "cosine_distance"
        ]
        best_score = min(scores) if scores else 1.0
        # En modo always_wait NUNCA se salta NLM: la respuesta siempre es híbrida.
        high_confidence = bool(
            hits and best_score < conf_thr and not wants_table and not always_wait
        )
        # Espera adaptativa (solo con NOTEBOOKLM_ALWAYS_WAIT=false): con evidencia
        # decente de Chroma se espera poco a NLM; sin evidencia, timeout completo;
        # las tablas necesitan a NLM sí o sí → presupuesto propio más amplio.
        decent = bool(hits and best_score < decent_thr and not wants_table)
        if always_wait:
            nlm_wait = hard_timeout
        elif wants_table:
            nlm_wait = float(getattr(settings, "NOTEBOOKLM_TABLE_TIMEOUT", 45.0))
        elif decent:
            nlm_wait = soft_timeout
        else:
            nlm_wait = parallel_timeout

        # Cachear la respuesta tardía/abandonada de NLM: el navegador sigue
        # trabajando y la siguiente pregunta igual o similar será cache-hit.
        def _cache_late(fut, _q=q, _nb=nb_id, _emb=q_emb):
            try:
                raw, _ = fut.result()
                if raw and raw.strip():
                    cache.set(_q, _nb, raw.strip(), [], embedding=_emb)
            except Exception:
                pass

        if cached:
            nlm_answer, _ = cached
            nlm_status = f"cache_hit_{cache_kind}"
        elif nlm_fut is not None:
            if high_confidence:
                if not nlm_fut.cancel():  # si ya corre, que termine y se cachee
                    nlm_fut.add_done_callback(_cache_late)
                nlm_status = "skipped_high_conf"
            else:
                try:
                    raw_answer, _ = nlm_fut.result(timeout=nlm_wait)
                    nlm_answer = (raw_answer or "").strip()
                    if nlm_answer:
                        cache.set(q, nb_id, nlm_answer, [], embedding=q_emb)
                    nlm_status = "ok"
                except FutureTimeoutError:
                    nlm_status = "timeout_soft" if decent else "timeout"
                    logger.warning(
                        "NotebookLM wait of %.0fs exceeded (decent_chroma=%s) — using Chroma only",
                        nlm_wait, decent,
                    )
                    nlm_fut.add_done_callback(_cache_late)
                except Exception as exc:
                    nlm_status = "error"
                    logger.warning("NotebookLM parallel query failed: %s", exc)

        # Routing:
        # - Table query + NLM answer → skip answer_node synthesis (preserves tabular format)
        # - Any other query with content → answer_node (LLM synthesis of both sources)
        # - Nothing found → no_context
        if wants_table and nlm_answer:
            route = "table_node"
        elif hits or nlm_answer:
            route = "answer_node"
        else:
            route = "no_context"

        if span is not None:
            try:
                span.update(output={
                    "hits_count": len(hits),
                    "best_chroma_score": round(best_score, 4) if scores else None,
                    "high_confidence": high_confidence,
                    "decent_chroma": decent,
                    "nlm_wait_budget_s": nlm_wait,
                    "wants_table": wants_table,
                    "nlm_answer_len": len(nlm_answer),
                    "nlm_status": nlm_status,
                    "route": route,
                })
            except Exception:
                pass

        out: GraphState = {"hits": hits, "nlm_answer": nlm_answer, "route": route}
        # For table_node, answer must be pre-loaded with the NLM response
        if route == "table_node":
            out["answer"] = nlm_answer
        return out


# Mensajes para el usuario cuando NotebookLM no entrega una respuesta útil.
_NLM_MSG_ERROR = (
    "⚠️ En este momento no pude procesar tu consulta. "
    "Por favor reformúlala con términos relacionados al RETIE "
    "(p. ej. instalaciones eléctricas, puesta a tierra, protecciones, tableros)."
)
_NLM_MSG_EMPTY = (
    "No encontré información sobre eso en el RETIE. "
    "¿Podrías reformular tu pregunta con más detalle?"
)
_NLM_MIN_QUERY_LEN = 4  # consultas más cortas casi nunca dan resultado y gastan ~30s


def _node_notebooklm(state: GraphState) -> GraphState:
    """Fallback retrieval via NotebookLM MCP when Chroma has no relevant hits.

    Distingue tres desenlaces para dar siempre un output claro al usuario:
      - ok:    respuesta válida de NotebookLM → continúa al stylist
      - error: fallo técnico (timeout del navegador, sesión, etc.) → mensaje para reformular
      - empty: NotebookLM respondió pero sin contenido útil → mensaje "sin información"
    """
    q = (state.get("question") or "").strip()

    # Rechazo rápido de consultas triviales para no gastar ~30s en el navegador.
    if len(q) < _NLM_MIN_QUERY_LEN:
        return {"answer": _NLM_MSG_EMPTY, "route": "deliver"}

    with span_ctx(None, "notebooklm_node", as_type="retriever", span_input={"question": q}) as span:
        status = "ok"
        answer = ""
        error_msg = ""
        try:
            client = _get_notebooklm_client()
            answer, _sources = client.ask_question(q, source_format="footnotes")
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            logger.warning("NotebookLM query failed: %s", exc)

        has_answer = bool(answer and len(answer.strip()) > 10)
        if not has_answer and status == "ok":
            status = "empty"

        if span is not None:
            try:
                span.update(
                    output={"answer_preview": answer[:200] if answer else error_msg[:200]},
                    metadata={"status": status},
                )
            except Exception:
                pass

        if has_answer:
            # Si el usuario pidió tabla, formatearla antes de entregar.
            route = "table_node" if state.get("wants_table") else "stylist_node"
            return {"answer": answer, "route": route}

        # Fallo técnico → pedir reformular; respuesta vacía → "sin información".
        # route="deliver" entrega el mensaje tal cual (no pasa por no_context,
        # que lo sobreescribiría con "No tengo evidencia en los documentos").
        user_msg = _NLM_MSG_ERROR if status == "error" else _NLM_MSG_EMPTY
        return {"answer": user_msg, "route": "deliver"}


# ===============================
#  Table Node (formato tabular)
# ===============================
import json as _json

# Tokens dedicados para el table_node. MAX_TOKENS general (p. ej. 400) trunca el
# JSON de tablas grandes; este límite alto evita que se corte a la mitad.
_TABLE_MAX_TOKENS = 4000

_TABLE_PROMPT = (
    "A partir de la INFORMACIÓN, extrae los datos en forma de tabla.\n"
    "Responde ÚNICAMENTE con JSON válido, sin texto adicional ni markdown, con esta forma:\n"
    '{{"title": "Título breve", "headers": ["Col1", "Col2"], "rows": [["a", "b"], ["c", "d"]]}}\n'
    "Reglas:\n"
    "- 'title' es un título corto y descriptivo en español (máx. 8 palabras).\n"
    "- Usa encabezados cortos y claros en español.\n"
    "- Extrae TODAS las filas de datos, sin resumir ni omitir ninguna.\n"
    "- Los fragmentos pueden venir desordenados o repetidos (solapamiento de\n"
    "  chunks): únelos, DEDUPLICA filas idénticas y ordénalas como en el documento.\n"
    "- Conserva los valores tal como aparecen, incluidos rangos ('26-30', '61 y más')\n"
    "  y fórmulas ('15 kW + 1 kW por cada estufa').\n"
    "- Cada fila debe tener exactamente el mismo número de celdas que headers.\n"
    "- NO incluyas notas, leyendas ni texto explicativo: solo encabezados y filas.\n"
    "- Mantén el JSON compacto, sin saltos de línea innecesarios.\n"
    "- Si la información NO es tabulable, responde {{\"headers\": [], \"rows\": []}}.\n\n"
    "Pregunta del usuario: {question}\n\n"
    "INFORMACIÓN:\n{answer}"
)


def _close_truncated_json(s: str) -> str:
    """Best-effort repair of JSON truncated mid-output (e.g. by a token limit).

    Balances open brackets/strings and drops a trailing comma so a cut-off
    rows array can still be parsed (the last, incomplete row is discarded)."""
    in_str = False
    esc = False
    stack: List[str] = []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()

    repaired = s
    if in_str:
        repaired += '"'
    # Drop a dangling comma / incomplete trailing token before closing.
    repaired = repaired.rstrip()
    while repaired and repaired[-1] in ",":
        repaired = repaired[:-1].rstrip()
    for opener in reversed(stack):
        repaired += "]" if opener == "[" else "}"
    return repaired


def _parse_table_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extrae y valida el JSON {headers, rows} de la salida del LLM.

    Tolera fences markdown, texto alrededor y truncamiento (token limit)."""
    if not raw:
        return None
    text = raw.strip()
    # Quitar fences ```json ... ``` si los hubiera
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:]
    end = candidate.rfind("}")
    snippet = candidate[: end + 1] if end != -1 else candidate

    data = None
    for attempt in (snippet, _close_truncated_json(candidate)):
        try:
            data = _json.loads(attempt)
            break
        except _json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return None

    headers = data.get("headers")
    rows = data.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list):
        return None
    if not headers or not rows:
        return None

    # Normalizar: solo filas que sean listas, recortadas/rellenadas a len(headers).
    ncol = len(headers)
    norm_rows: List[List[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))
        norm_rows.append(row[:ncol])
    if not norm_rows:
        return None

    title = data.get("title")
    return {
        "title": title if isinstance(title, str) else None,
        "headers": headers,
        "rows": norm_rows,
    }


_TABLE_PARTIAL_NOTE = (
    "⚠️ No pude reconstruir la tabla completa en este intento "
    "(la fuente NotebookLM no estuvo disponible). "
    "Esta es la información encontrada en la base local:\n\n"
)

# Tope de caracteres de fragmentos crudos que se anexan a la extracción de tabla
# (~12k chars ≈ 3k tokens; suficiente para TOP_K_TABLES chunks de 800 chars).
_TABLE_RAW_CONTEXT_CHARS = 12000


def _node_table(state: GraphState) -> GraphState:
    """Convierte la respuesta del agente en una tabla y la entrega como imagen PNG.

    Pide al LLM estructurar los datos como JSON {title, headers, rows} y renderiza
    una imagen (sin el botón "COPIAR CÓDIGO" que Telegram añade a los <pre>).
    La extracción usa la respuesta sintetizada Y los fragmentos crudos del
    retriever: la síntesis comprime/omite filas, los chunks originales no.
    Degradación elegante:
      - Si Pillow/imagen falla → tabla de texto adaptativa (<pre>).
      - Si la información no es tabulable o el LLM falla → prosa original
        (con nota de tabla parcial cuando NotebookLM no estuvo disponible).
    """
    base_answer = state.get("answer", "") or ""
    question = state.get("question", "")
    if not base_answer.strip():
        return {"answer": base_answer, "route": "stylist_node"}

    # Fragmentos crudos: la fuente más fiel para no omitir filas. La respuesta
    # del answer_node ya pasó por un LLM con límite de tokens y puede haber
    # resumido; los chunks de Chroma traen el texto literal del PDF.
    raw_chunks = "\n\n".join(
        h.get("text", "") for h in (state.get("hits") or []) if h.get("text")
    )[:_TABLE_RAW_CONTEXT_CHARS]
    info = base_answer
    if raw_chunks:
        info = f"{base_answer}\n\nFRAGMENTOS LITERALES DEL DOCUMENTO:\n{raw_chunks}"

    model = _resolve_model(state.get("agent_key"), explicit=None)
    prompt = _TABLE_PROMPT.format(question=question, answer=info)

    with span_ctx(
        None, "table_node",
        {"model": model},
        as_type="chain",
        span_input={"question": question},
    ) as span:
        status = "ok"
        final_text = base_answer
        image: Optional[bytes] = None
        try:
            resp = _client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=_TABLE_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "Eres un formateador de datos a tablas. Solo devuelves JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            table = _parse_table_json(raw)
            if not table:
                # Diagnóstico: deja rastro del JSON que no se pudo parsear.
                logger.warning("table_node: no se pudo parsear JSON (%d chars): %s",
                               len(raw), raw[:300])
            if table:
                title = table.get("title")
                try:
                    image = render_table_image(table["headers"], table["rows"], title=title)
                    # Caption breve para acompañar la imagen.
                    final_text = f"📊 {title}" if title else "📊 Tabla solicitada"
                except Exception as img_exc:
                    # Sin Pillow / sin fuente → degradar a tabla de texto.
                    status = "text_fallback"
                    logger.warning("table image failed, using text table: %s", img_exc)
                    final_text = render_telegram_table(table["headers"], table["rows"])
            else:
                status = "not_tabular"  # se conserva la prosa original
                # Honestidad con el usuario: si NLM no aportó (fuente clave para
                # tablas) y no se pudo tabular, avisar que el resultado es parcial.
                if not (state.get("nlm_answer") or "").strip():
                    final_text = _TABLE_PARTIAL_NOTE + base_answer
        except Exception as exc:
            status = "error"
            logger.warning("table_node failed: %s", exc)
            if not (state.get("nlm_answer") or "").strip():
                final_text = _TABLE_PARTIAL_NOTE + base_answer

        if span is not None:
            try:
                span.update(
                    output={"preview": final_text[:200], "has_image": image is not None},
                    metadata={"status": status},
                )
            except Exception:
                pass

        try:
            log_generation(
                None,
                name="table_node",
                input_text=base_answer,
                output_text=final_text,
                model=model,
                metadata={"status": status, "has_image": image is not None},
            )
        except Exception:
            pass

    out: GraphState = {"answer": final_text, "route": "stylist_node"}
    if image is not None:
        out["table_image"] = image
    return out


from retie_agent.agent.enrichment_assistant import EnrichmentAssistant
import os
from retie_agent.config import ENRICHMENT_ASSISTANT_ID, ENRICHMENT_VECTOR_STORE_ID
from retie_agent.observability.obs import span_ctx, log_generation  # ✅ keep logs visible in Langfuse

# ===============================
#  Enrichment Node
# ===============================
# Lazy: instanciar EnrichmentAssistant en import rompía el arranque cuando no
# había OPENAI_API_KEY en el entorno (p. ej. tests o tooling local).
_enrichment_agent: Optional[EnrichmentAssistant] = None


def _get_enrichment_agent() -> EnrichmentAssistant:
    global _enrichment_agent
    if _enrichment_agent is None:
        _enrichment_agent = EnrichmentAssistant(
            assistant_id=ENRICHMENT_ASSISTANT_ID,
            vector_store_id=ENRICHMENT_VECTOR_STORE_ID,
        )
    return _enrichment_agent


def _node_enrich(state: GraphState) -> GraphState:
    """Asistente secundario que enriquece la respuesta base sin bloquear el flujo.

    Con ENRICHMENT_ENABLED=false el nodo es un pass-through (respuesta ~2x más
    rápida). Si el assistant falla o excede ENRICHMENT_TIMEOUT, se conserva la
    respuesta base SIN exponer el error interno al usuario.
    """
    base_answer = state.get("answer", "")
    question = state.get("question", "")

    if not as_bool(getattr(settings, "ENRICHMENT_ENABLED", "true")):
        return {"answer": base_answer}

    timeout = float(getattr(settings, "ENRICHMENT_TIMEOUT", 30.0))

    with span_ctx(
        None,
        "enrich_node",
        {"assistant_id": ENRICHMENT_ASSISTANT_ID},
        as_type="chain",
    ):
        try:
            fut = _POOL.submit(
                _get_enrichment_agent().enrich_response,
                user_message=question,
                draft_response=base_answer,
            )
            enriched = fut.result(timeout=timeout)

            # Si el enriquecimiento no produce texto, usa la respuesta base
            if not enriched or len(enriched.strip()) < 20:
                raise ValueError("Sin resultados de enriquecimiento o vector vacío")

            status = "ok"

        except Exception as e:
            # Falla técnica → conservar la respuesta original tal cual.
            # (Antes se anexaba el error al texto y el usuario lo veía.)
            logger.warning("enrich_node fallback (%s)", e)
            enriched = base_answer
            status = "fallback"

        # 🔍 Registro en Langfuse: permite ver la salida y el estado del enriquecimiento
        try:
            log_generation(
                None,
                name="enrich_node",
                input_text=question,
                output_text=enriched,
                model="assistant_enrichment",
                metadata={
                    "assistant_id": ENRICHMENT_ASSISTANT_ID,
                    "vector_store_id": ENRICHMENT_VECTOR_STORE_ID,
                    "status": status,
                },
            )
        except Exception:
            pass

        return {"answer": enriched}


# ===============================
#  Stylist Node (Output Formatter)
# ===============================
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_stylist(state: GraphState) -> GraphState:
    """
    Post-processing node that prepares the enriched response for downstream delivery.

    It can adapt tone or format depending on the output channel (Telegram, WhatsApp, Web, etc.).
    The output is a JSON-ready structure, making it easy to add or swap channels later.
    """
    enriched_answer = state.get("answer", "")
    question = state.get("question", "")
    hits = state.get("hits", []) or []
    metadata = state.get("metadata", {}) or {}
    channel = metadata.get("channel", "telegram")  # default channel
    table_image = state.get("table_image")  # PNG bytes if the answer is a table

    # Citas a partir de los documentos realmente recuperados (no del LLM).
    sources, sources_text = _build_user_sources(hits)

    with span_ctx(
        None,
        "stylist_node",
        {"channel": channel},
        as_type="chain",
    ):
        try:
            # Simple rules: add emojis, Markdown, or remove unsupported tags depending on channel
            if channel == "telegram":
                styled_text = enriched_answer  # Telegram already supports Markdown/HTML
            elif channel == "whatsapp":
                styled_text = enriched_answer.replace("*", "").replace("_", "")
            elif channel == "web":
                styled_text = f"<p>{enriched_answer}</p>"
            else:
                styled_text = enriched_answer

            # Unified JSON payload (future-proof)
            styled_payload = {
                "channel": channel,
                "user_message": question,
                "formatted_response": styled_text,
                "sources": sources,
                "sources_text": sources_text,
                "image": table_image,  # PNG bytes → router sends it as a photo
                "timestamp": _utcnow_iso(),
            }

            status = "ok"

        except Exception as e:
            styled_payload = {
                "channel": channel,
                "user_message": question,
                "formatted_response": enriched_answer,
                "sources": sources,
                "sources_text": sources_text,
                "image": table_image,
                "error": str(e),
                "timestamp": _utcnow_iso(),
            }
            status = "fallback"

        # Log to Langfuse for visibility (sin los bytes crudos de la imagen)
        try:
            loggable = {k: v for k, v in styled_payload.items() if k != "image"}
            loggable["image"] = f"<{len(table_image)} bytes PNG>" if table_image else None
            log_generation(
                None,
                name="stylist_node",
                input_text=question,
                output_text=str(loggable),
                model="stylist_formatter",
                metadata={"channel": channel, "status": status, "has_image": table_image is not None},
            )
        except Exception:
            pass

        # return both final text and structured JSON for downstream consumers
        return {"answer": styled_payload}


# ---------- Graph builder ----------
@dataclass
class _Compiled:
    app: Any

_COMPILED: Optional[_Compiled] = None

def build_graph():
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED.app

    g = StateGraph(GraphState)
    g.add_node("route_entry", _node_route_entry)
    g.add_node("condense_node", _node_condense)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("router", _node_router)
    g.add_node("hybrid_retrieve", _node_hybrid_retrieve)
    g.add_node("answer_node", _node_answer)
    g.add_node("enrich_node", _node_enrich)
    g.add_node("stylist_node", _node_stylist)
    g.add_node("no_context", _node_no_context)
    g.add_node("table_node", _node_table)
    g.add_node("smalltalk_node", _node_smalltalk)

    # Entry: smalltalk responde directo; lo demás pasa por condense_node
    # (reescritura de seguimiento) y luego hybrid (Chroma+NLM) o Chroma-only.
    g.set_entry_point("route_entry")
    g.add_conditional_edges(
        "route_entry",
        lambda s: "smalltalk" if s.get("route") == "smalltalk" else "condense",
        {"smalltalk": "smalltalk_node", "condense": "condense_node"},
    )
    g.add_edge("smalltalk_node", "stylist_node")
    g.add_conditional_edges(
        "condense_node",
        lambda s: s.get("route", "retrieve"),
        {"retrieve": "retrieve", "hybrid_retrieve": "hybrid_retrieve"},
    )

    # Chroma-only path: retrieve → router → answer → (table | enrich) → stylist → end
    g.add_edge("retrieve", "router")
    g.add_conditional_edges(
        "router",
        lambda s: s.get("route", "no_context"),
        {"answer_node": "answer_node", "no_context": "no_context"},
    )

    # Hybrid path:
    #   table query + NLM answer → table_node (skips answer_node to preserve tabular format)
    #   any content found        → answer_node (LLM synthesis of Chroma + NLM)
    #   nothing found            → no_context
    g.add_conditional_edges(
        "hybrid_retrieve",
        lambda s: s.get("route", "no_context"),
        {"answer_node": "answer_node", "no_context": "no_context", "table_node": "table_node"},
    )

    # Shared answer path: answer_node → (table | enrich) → stylist → end
    g.add_conditional_edges(
        "answer_node",
        lambda s: "table_node" if s.get("wants_table") else "enrich_node",
        {"table_node": "table_node", "enrich_node": "enrich_node"},
    )
    g.add_edge("enrich_node", "stylist_node")
    g.add_edge("table_node", "stylist_node")
    g.add_edge("stylist_node", END)
    g.add_edge("no_context", END)

    app = g.compile()
    _COMPILED = _Compiled(app=app)
    return app


# ---------- Runner ----------
def _make_langfuse_handler():
    """Crea CallbackHandler solo si Langfuse está habilitado, con credenciales explícitas."""
    enabled = str(getattr(settings, "LANGFUSE_ENABLED", "false")).lower() in ("1", "true", "yes")
    if not enabled:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        kwargs: dict = {}
        pk = getattr(settings, "LANGFUSE_PUBLIC_KEY", None) or os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = getattr(settings, "LANGFUSE_SECRET_KEY", None) or os.getenv("LANGFUSE_SECRET_KEY")
        host = (
            getattr(settings, "LANGFUSE_HOST", None)
            or os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
        )
        if pk:
            kwargs["public_key"] = pk
        if sk:
            kwargs["secret_key"] = sk
        if host:
            kwargs["host"] = host
        return CallbackHandler(**kwargs)
    except Exception:
        return None


def run_graph(
    question: str,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    app = build_graph()

    langfuse_handler = _make_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    history = get_history(session, limit=getattr(settings, "HISTORY_LIMIT", 10))
    add_message(session, user_id, "user", question)

    state_in = make_state(
        question,
        user_id=user_id,
        session=session,
        agent_key=agent_key,
        history=history,
    )

    trace_input = {
        "question": question,
        "user_id": user_id,
        "session": session,
        "agent_key": agent_key,
        "history": history,
    }

    with trace_ctx(
        name="retie-query",
        user_id=user_id,
        metadata={"agent_key": agent_key or "default", **(metadata or {})},
        trace_input=trace_input,
    ) as trace_span:
        result: Dict[str, Any] = app.invoke(
            state_in,
            config={
                "callbacks": callbacks,
                "run_name": "LangGraph",
                "tags": ["retie-agent", "graph", f"user:{user_id}", f"session:{session}"],
                "metadata": {**(metadata or {}), "agent_key": agent_key or "default"},
            },
        )

        final_answer = result.get("answer") if isinstance(result, dict) else None
        final_txt = final_answer or "No tengo evidencia en los documentos."
        output_txt = final_txt.get("formatted_response", str(final_txt)) if isinstance(final_txt, dict) else str(final_txt)

        if trace_span is not None:
            try:
                trace_span.set_trace_io(output=output_txt)
            except Exception:
                pass

    txt_to_save = final_txt
    if isinstance(final_txt, dict) and "formatted_response" in final_txt:
        txt_to_save = final_txt["formatted_response"]
    add_message(session, user_id, "assistant", str(txt_to_save))

    return final_txt
