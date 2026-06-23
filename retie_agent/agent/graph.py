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
from retie_agent.retriever.retrieve import search, expand_hits_with_page_context
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
from retie_agent.agent.notebooklm_client import NotebookLMClient, NotebookLMError, NotebookLMCache, NLM_RECOVERY_NEEDED
from retie_agent.agent.table_render import render_telegram_table, render_table_image
from retie_agent.agent.intent import (
    classify_intent,
    _TABLE_RE, _FULL_RE, _SMALLTALK_RE,   # re-exportados para compatibilidad
    _wants_table, _wants_full,
)

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
    # Salidas SEPARADAS de cada fuente de recuperación. Cada rama paralela escribe
    # en su propia clave (sin solaparse) para que la traza de Langfuse muestre, al
    # hacer clic en chromadb_node / notebooklm_node, exactamente lo que aportó esa
    # fuente. answer_node hace el fan-in: lee ambas y deriva `hits`/`nlm_answer`.
    chromadb_docs: List[Dict[str, Any]]   # documentos recuperados de ChromaDB
    notebooklm_docs: Optional[str]        # respuesta de NotebookLM
    hits: List[Dict[str, Any]]            # = chromadb_docs (lo consumen table/stylist)
    route: str
    answer: Any   # may now hold dict for styled payload
    metadata: Optional[Dict[str, Any]]  # new, for passing channel info
    history: List[Dict[str, str]]
    wants_table: bool       # user asked for a "tabla" → route through table_node
    wants_full: bool        # user asked for an exhaustive answer (todos los numerales…)
    table_image: Any        # PNG bytes when the table is rendered as an image
    nlm_answer: Optional[str]  # = notebooklm_docs (derivado en answer_node para table_node)
    notebooklm_enabled: bool   # flag que activa/desactiva el fan-out a notebooklm_node
    search_query: Optional[str]  # standalone query (condensed from history) used for retrieval
    suggestions: List[str]  # follow-up questions offered to the user after the answer
    intent: Optional[str]        # exhaustiva | tabla | puntual | smalltalk (TICKET-001)
    intent_source: Optional[str]  # "regex" | "llm" — origen de la clasificación


# La detección de intención (tabla / exhaustiva / smalltalk) vive en
# retie_agent/agent/intent.py: clasificador LLM barato + regex como fallback
# (_TABLE_RE, _FULL_RE, _SMALLTALK_RE, _wants_table, _wants_full se importan arriba).
# route_entry usa classify_intent() como fuente primaria de decisión.

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


def _format_nlm_sources(sources: List[Any], limit: int = 20) -> List[Dict[str, Any]]:
    """Normaliza las fuentes/citas (footnotes) que devuelve NotebookLM para que
    sean legibles en Langfuse, igual que _format_chunks_for_trace hace con Chroma.
    La estructura real varía (items MCP 'resource' o dicts de 'data.sources'), así
    que se extraen campos comunes de forma defensiva."""
    out: List[Dict[str, Any]] = []
    for i, s in enumerate((sources or [])[:limit], start=1):
        if isinstance(s, dict):
            text = s.get("text")
            item: Dict[str, Any] = {
                "rank": i,
                "title": s.get("title") or s.get("name") or s.get("label"),
                "uri": s.get("uri") or s.get("url") or s.get("source"),
                "type": s.get("type"),
            }
            if isinstance(text, str) and text:
                item["text"] = text[:500]
            out.append({k: v for k, v in item.items() if v is not None})
        else:
            out.append({"rank": i, "value": str(s)[:500]})
    return out


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


def _node_chromadb(state: GraphState) -> GraphState:
    """Rama de recuperación contra ChromaDB (paralela a notebooklm_node).

    Nodo independiente: escribe su salida en `chromadb_docs` para que, en la
    traza de Langfuse, al hacer clic en chromadb_node se vean exactamente los
    documentos/scores recuperados de Chroma — separados de lo que aportó NLM.
    """
    q = state.get("search_query") or state["question"]
    agent_key = state.get("agent_key")
    coll = _resolve_collection(agent_key, explicit=None)
    # Tablas y respuestas exhaustivas: más chunks — el contenido largo
    # (tablas, artículos con literales a–y) vive partido en varios fragmentos.
    needs_breadth = state.get("wants_table") or state.get("wants_full")
    top_k = (
        getattr(settings, "TOP_K_TABLES", 12)
        if needs_breadth
        else getattr(settings, "TOP_K", 4)
    )
    thr = getattr(settings, "RAG_DISTANCE_THRESHOLD", 0.45)

    span_input = {
        "question": q,
        "collection": coll,
        "top_k": top_k,
        "distance_threshold": thr,
    }
    with span_ctx(None, "chromadb_node", as_type="retriever", span_input=span_input) as span:
        try:
            raw_hits = search(q, top_k=top_k, collection_name=coll)
            # Tablas/exhaustivas: NO deduplicar por (source, page) — una página
            # suele tener varios chunks y todos pueden contener filas o literales.
            # Se expande además a las páginas completas de los mejores hits.
            if needs_breadth:
                hits = expand_hits_with_page_context(raw_hits)
            else:
                hits = _dedupe_hits(raw_hits)
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
        return {"chromadb_docs": hits}

def _node_route_entry(state: GraphState) -> GraphState:
    """Entry point: smalltalk directo, o recuperación (Chroma ± NLM) en el resto.

    El bifurcado activo/desactivado de NotebookLM NO se decide aquí: se publica el
    flag `notebooklm_enabled` en el estado y la conditional edge desde condense_node
    (`_fanout_retrieval`) lo usa para incluir o no a notebooklm_node en el fan-out.
    """
    q = state.get("question", "")
    nlm_enabled = str(getattr(settings, "NOTEBOOKLM_ENABLED", "false")).lower() in ("1", "true", "yes")
    # Clasificador de intención (TICKET-001): LLM barato con fallback a regex.
    # Reemplaza el ruteo por regex como fuente PRIMARIA de la decisión.
    intent = classify_intent(q, history=state.get("history"))
    route = intent.route
    wants_table = intent.wants_table
    wants_full = intent.wants_full
    with span_ctx(
        None, "route_entry", as_type="chain",
        span_input={
            "nlm_enabled": nlm_enabled,
            "intent": intent.intent,
            "intent_source": intent.source,
            "wants_table": wants_table,
            "wants_full": wants_full,
        },
    ) as span:
        if span is not None:
            try:
                span.update(output={
                    "route": route,
                    "intent": intent.intent,
                    "intent_source": intent.source,
                    "wants_table": wants_table,
                    "wants_full": wants_full,
                })
            except Exception:
                pass
    return {
        "route": route,
        "wants_table": wants_table,
        "wants_full": wants_full,
        "intent": intent.intent,
        "intent_source": intent.source,
        "notebooklm_enabled": nlm_enabled,
    }


def _node_answer(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")
    # Fan-in de las dos ramas de recuperación: cada nodo escribió en su propia
    # clave (chromadb_docs / notebooklm_docs) sin solaparse. Aquí se combinan para
    # el LLM y se derivan `hits`/`nlm_answer`, que consumen los nodos posteriores
    # (table_node, stylist_node) tal como antes.
    hits = state.get("chromadb_docs") or []
    nlm_answer = (state.get("notebooklm_docs") or "").strip()

    if not hits and not nlm_answer:
        return {
            "answer": "No tengo evidencia en los documentos.",
            "hits": hits,
            "nlm_answer": nlm_answer,
            "route": "no_context",
        }

    model = _resolve_model(agent_key, explicit=None)
    sys = SYSTEM_PROMPT_RETIE
    # Use hybrid prompt when both sources are available
    prompt = make_hybrid_prompt(hits, nlm_answer, q, is_admin=False)
    messages = [{"role": "system", "content": sys}]
    for msg in state.get("history", []):
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})

    temperature = 0.0
    # Tablas y respuestas exhaustivas (todos los numerales a–y) necesitan mucho
    # más presupuesto de salida; las respuestas normales conservan el default.
    wants_table = state.get("wants_table", False)
    wants_full = state.get("wants_full", False)
    default_max = getattr(settings, "MAX_TOKENS", 600)
    max_tokens = 4000 if (wants_table or wants_full) else default_max

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

        # `hits`/`nlm_answer` se exponen para los nodos posteriores (table_node usa
        # los chunks crudos y la nota de "tabla parcial"; stylist_node arma las citas).
        return {"answer": answer, "hits": hits, "nlm_answer": nlm_answer, "route": "answer"}

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
#  NotebookLM Node (rama independiente, paralela a chromadb_node)
# ===============================
_nlm_cache: Optional[NotebookLMCache] = None


def _get_nlm_cache() -> NotebookLMCache:
    global _nlm_cache
    if _nlm_cache is None:
        ttl = int(getattr(settings, "NOTEBOOKLM_CACHE_TTL", 3600))
        _nlm_cache = NotebookLMCache(ttl=ttl)
    return _nlm_cache


def _node_notebooklm(state: GraphState) -> GraphState:
    """Rama de recuperación contra NotebookLM (paralela a chromadb_node).

    Nodo independiente: escribe su salida en `notebooklm_docs` para que, en la
    traza de Langfuse, al hacer clic en notebooklm_node se vea EXACTAMENTE lo que
    aportó NotebookLM — separado de lo recuperado por Chroma. Solo se ejecuta
    cuando el fan-out condicional lo incluye (NOTEBOOKLM_ENABLED=true); con el flag
    apagado el nodo ni siquiera aparece en la traza.

    Conserva el caché (exacto + semántico) y un tope de espera
    (`NOTEBOOKLM_HARD_TIMEOUT`) para que un navegador colgado no bloquee el fan-in
    hacia answer_node. La respuesta tardía/abandonada se cachea en segundo plano
    para que la siguiente pregunta igual o similar sea cache-hit.

    Nota: al ser una rama paralela, este nodo NO ve los resultados de Chroma, así
    que el atajo de "alta confianza de Chroma → saltar NLM" del antiguo
    hybrid_retrieve ya no aplica (coincide con el modo por defecto
    NOTEBOOKLM_ALWAYS_WAIT=true, que nunca saltaba NLM).
    """
    q = state.get("search_query") or state["question"]
    nb_id = getattr(settings, "NOTEBOOKLM_NOTEBOOK_ID", None)
    min_chars = int(getattr(settings, "NOTEBOOKLM_MIN_QUERY_CHARS", 12))
    sem_sim = float(getattr(settings, "NLM_SEMANTIC_CACHE_SIM", 0.93))
    hard_timeout = float(getattr(settings, "NOTEBOOKLM_HARD_TIMEOUT", 180.0))

    nlm_eligible = len(q.strip()) >= min_chars
    cache = _get_nlm_cache()
    cache_kind = None
    cached = cache.get(q, nb_id) if nlm_eligible else None
    if cached:
        cache_kind = "exact"

    # Caché semántico: una pregunta reformulada pero equivalente reutiliza la
    # respuesta NLM previa. El embedding se comparte vía LRU con el retrieval de
    # Chroma (no hay llamada extra a la API en el camino feliz).
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
        None, "notebooklm_node", as_type="retriever",
        span_input={"question": q, "nlm_cache_hit": cache_kind},
    ) as span:
        nlm_answer = ""
        nlm_sources: List[Dict[str, Any]] = []
        error_msg = ""
        status = "disabled"

        # Cachear la respuesta tardía/abandonada de NLM: el navegador sigue
        # trabajando y la siguiente pregunta igual o similar será cache-hit.
        def _cache_late(fut, _q=q, _nb=nb_id, _emb=q_emb):
            try:
                raw, raw_src = fut.result()
                if raw and raw.strip():
                    cache.set(_q, _nb, raw.strip(), raw_src or [], embedding=_emb)
            except Exception:
                pass

        if cached:
            nlm_answer = (cached[0] or "").strip()
            nlm_sources = cached[1] or []
            status = f"cache_hit_{cache_kind}"
        elif not nlm_eligible:
            status = "skipped_short_query"
        else:
            # Pool persistente: si se agota la espera, el future queda corriendo en
            # background (se cachea vía callback) sin bloquear el fan-in.
            client = _get_notebooklm_client()
            nlm_fut = _POOL.submit(client.ask_question, q, "footnotes", nb_id)
            try:
                raw_answer, raw_sources = nlm_fut.result(timeout=hard_timeout)
                nlm_answer = (raw_answer or "").strip()
                nlm_sources = raw_sources or []
                if nlm_answer:
                    cache.set(q, nb_id, nlm_answer, nlm_sources, embedding=q_emb)
                # "empty" = NotebookLM respondió pero sin texto (sesión no
                # autenticada, notebook_id inválido o sin coincidencias).
                status = "ok" if nlm_answer else "empty"
            except FutureTimeoutError:
                status = "timeout"
                error_msg = f"NotebookLM no respondió en {hard_timeout:.0f}s"
                logger.warning(
                    "notebooklm_node: espera de %.0fs agotada — sin aporte de NLM",
                    hard_timeout,
                )
                nlm_fut.add_done_callback(_cache_late)
            except Exception as exc:
                status = "error"
                error_msg = str(exc)
                logger.warning("notebooklm_node query failed: %s", exc)
                _msg = error_msg.lower()
                if any(k in _msg for k in ("authenticated", "not auth", "login", "google", "expired", "cookie")):
                    NLM_RECOVERY_NEEDED.set()
                    logger.warning("notebooklm_node: sesión posiblemente expirada — recuperación on-demand programada")

        if span is not None:
            try:
                # Output rico para depurar QUÉ trae NLM (y por qué viene vacío):
                # el texto, las citas/footnotes, el status y el error si lo hubo.
                output: Dict[str, Any] = {
                    "answer": nlm_answer,
                    "answer_len": len(nlm_answer),
                    "answer_preview": nlm_answer[:1000],
                    "sources": _format_nlm_sources(nlm_sources),
                    "sources_count": len(nlm_sources),
                    "status": status,
                }
                if error_msg:
                    output["error"] = error_msg[:800]
                span.update(
                    output=output,
                    metadata={
                        "status": status,
                        "nlm_cache_hit": cache_kind,
                        "notebook_id": nb_id,
                        "eligible": nlm_eligible,
                        "wait_budget_s": hard_timeout,
                        "question": q,
                    },
                )
            except Exception:
                pass

        return {"notebooklm_docs": nlm_answer}


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
# (~24k chars ≈ 6k tokens; cubre TOP_K_TABLES chunks + expansión de páginas).
_TABLE_RAW_CONTEXT_CHARS = 24000


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
    # Orden de documento (source, página) con sort estable: los chunks de una
    # misma página conservan su orden de inserción (ids secuenciales).
    _hits_sorted = sorted(
        (h for h in (state.get("hits") or []) if h.get("text")),
        key=lambda h: (
            str((h.get("meta") or {}).get("source", "")),
            (h.get("meta") or {}).get("page", 0) or 0,
        ),
    )
    raw_chunks = "\n\n".join(h.get("text", "") for h in _hits_sorted)[:_TABLE_RAW_CONTEXT_CHARS]
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
from retie_agent.agent.fidelity import enrichment_preserves_fidelity, fidelity_diff
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

        # 🛡️ Guarda de fidelidad (TICKET-003 / H-903): el enriquecedor solo puede
        # mejorar la redacción, NO alterar números, referencias ni listas. Si la
        # versión enriquecida cambió, omitió o inventó algún token crítico respecto
        # del borrador, se DESCARTA y se entrega el borrador (siempre fiel).
        fidelity_drift = None
        if status == "ok" and not enrichment_preserves_fidelity(base_answer, enriched):
            fidelity_drift = fidelity_diff(base_answer, enriched)
            logger.warning("enrich_node rechazado por deriva de fidelidad: %s", fidelity_drift)
            enriched = base_answer
            status = "rejected_fidelity"

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
                    "fidelity_drift": fidelity_drift,
                },
            )
        except Exception:
            pass

        return {"answer": enriched}


# ===============================
#  Suggestion Node (preguntas de seguimiento)
# ===============================
_SUGGEST_PROMPT = (
    "Eres un asistente experto en normativa eléctrica colombiana (RETIE y NTC 2050).\n"
    "Con base en los últimos mensajes del usuario y la respuesta que recibió, "
    "genera {n} preguntas de seguimiento CORTAS (máx. 12 palabras cada una) que "
    "el usuario probablemente quiera hacer a continuación sobre el mismo tema.\n"
    "Reglas:\n"
    "- Preguntas concretas y respondibles con el RETIE o la NTC 2050.\n"
    "- No repitas preguntas que el usuario ya hizo.\n"
    "- Responde ÚNICAMENTE con un array JSON de strings, sin texto adicional.\n"
    'Ejemplo: ["¿Qué calibre de conductor exige la NTC 2050 para 40 A?"]\n\n'
    "ÚLTIMOS MENSAJES DEL USUARIO:\n{user_messages}\n\n"
    "RESPUESTA DADA (resumen):\n{answer}\n"
)


def _parse_suggestions(raw: str, limit: int) -> List[str]:
    """Extrae el array JSON de la salida del LLM (tolerante a fences/texto)."""
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = _json.loads(text[start:end + 1])
    except _json.JSONDecodeError:
        return []
    out = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    return out[:limit]


def _node_suggest(state: GraphState) -> GraphState:
    """Genera preguntas de seguimiento basadas en los últimos mensajes del usuario.

    Falla en silencio: cualquier error deja suggestions=[] y la respuesta sale igual.
    """
    if not as_bool(getattr(settings, "SUGGESTIONS_ENABLED", "true")):
        return {"suggestions": []}

    n = max(1, int(getattr(settings, "SUGGESTIONS_COUNT", 3)))
    # Últimos 3 mensajes del usuario (historial) + la pregunta actual.
    user_msgs = [m.get("content", "") for m in (state.get("history") or []) if m.get("role") == "user"]
    user_msgs = [m for m in user_msgs if m][-2:] + [state.get("question", "")]
    answer = state.get("answer", "")
    answer_text = answer.get("formatted_response", "") if isinstance(answer, dict) else str(answer or "")

    with span_ctx(
        None, "suggest_node", as_type="chain",
        span_input={"user_messages": user_msgs},
    ) as span:
        suggestions: List[str] = []
        try:
            prompt = _SUGGEST_PROMPT.format(
                n=n,
                user_messages="\n".join(f"- {m[:300]}" for m in user_msgs),
                answer=answer_text[:600],
            )
            resp = _client.chat.completions.create(
                model=_resolve_model(state.get("agent_key"), explicit=None),
                temperature=0.7,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            suggestions = _parse_suggestions(resp.choices[0].message.content or "", n)
        except Exception as exc:
            logger.warning("suggest_node failed: %s", exc)

        if span is not None:
            try:
                span.update(output={"suggestions": suggestions})
            except Exception:
                pass
        return {"suggestions": suggestions}


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
                "suggestions": state.get("suggestions") or [],
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
                "suggestions": state.get("suggestions") or [],
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
    # Ramas de recuperación INDEPENDIENTES → cada una genera su propio span en
    # Langfuse (chromadb_node vs notebooklm_node), con su salida en clave propia.
    g.add_node("chromadb_node", _node_chromadb)
    g.add_node("notebooklm_node", _node_notebooklm)
    g.add_node("answer_node", _node_answer)
    g.add_node("enrich_node", _node_enrich)
    g.add_node("stylist_node", _node_stylist)
    g.add_node("no_context", _node_no_context)
    g.add_node("table_node", _node_table)
    g.add_node("smalltalk_node", _node_smalltalk)
    g.add_node("suggest_node", _node_suggest)

    # Entry: smalltalk responde directo; lo demás pasa por condense_node
    # (reescritura de seguimiento) y de ahí al fan-out de recuperación.
    g.set_entry_point("route_entry")
    g.add_conditional_edges(
        "route_entry",
        lambda s: "smalltalk" if s.get("route") == "smalltalk" else "condense",
        {"smalltalk": "smalltalk_node", "condense": "condense_node"},
    )
    g.add_edge("smalltalk_node", "stylist_node")

    # ── Fan-out de recuperación (ramas paralelas que convergen) ──────────────
    # La conditional edge devuelve la LISTA de ramas a ejecutar:
    #   - NotebookLM ACTIVO    → ["chromadb_node", "notebooklm_node"]  (paralelo)
    #   - NotebookLM DESACTIVADO → ["chromadb_node"]  (notebooklm_node no se ejecuta
    #                              ni aparece en la traza)
    # LangGraph corre ambas ramas en el mismo super-step y espera a que terminen
    # (fan-in) antes de ejecutar answer_node.
    def _fanout_retrieval(s: GraphState) -> List[str]:
        targets = ["chromadb_node"]
        if s.get("notebooklm_enabled"):
            targets.append("notebooklm_node")
        return targets

    g.add_conditional_edges(
        "condense_node",
        _fanout_retrieval,
        {"chromadb_node": "chromadb_node", "notebooklm_node": "notebooklm_node"},
    )

    # Fan-in: ambas ramas convergen en answer_node. LangGraph lo ejecuta una sola
    # vez, tras completarse las ramas que se hayan activado en el fan-out.
    g.add_edge("chromadb_node", "answer_node")
    g.add_edge("notebooklm_node", "answer_node")

    # answer_node combina ambas fuentes y enruta:
    #   - sin evidencia en ninguna fuente → no_context
    #   - tabla solicitada                → table_node
    #   - respuesta exhaustiva (wants_full) salta el enriquecedor (recortaría numerales)
    #   - resto                           → enrich_node
    def _after_answer(s: GraphState) -> str:
        if s.get("route") == "no_context":
            return "no_context"
        if s.get("wants_table"):
            return "table_node"
        if s.get("wants_full"):
            return "suggest_node"
        return "enrich_node"

    g.add_conditional_edges(
        "answer_node",
        _after_answer,
        {
            "no_context": "no_context",
            "table_node": "table_node",
            "enrich_node": "enrich_node",
            "suggest_node": "suggest_node",
        },
    )
    # Tras enriquecer/tabular se generan sugerencias de seguimiento y se estiliza.
    g.add_edge("enrich_node", "suggest_node")
    g.add_edge("table_node", "suggest_node")
    g.add_edge("suggest_node", "stylist_node")
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
