# app/agent/graph.py
from __future__ import annotations
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from openai import OpenAI

from retie_agent.config import settings, as_bool
from retie_agent.llm.provider import create_chat_completion
from retie_agent.retriever.retrieve import search, expand_hits_with_page_context
from retie_agent.agent.prompt import (
    make_prompt,
    make_hybrid_prompt,
    SYSTEM_PROMPT_RETIE,
    SYSTEM_PROMPT_MEDIA,
    QUERY_ENRICHMENT_PROMPT,
    format_history_for_condense,
)
from retie_agent.agent.retie_agent import _dedupe_hits, _resolve_collection, _resolve_model
from retie_agent.observability.obs import trace_ctx, span_ctx, log_generation
from retie_agent.services.history import get_history, add_message
from retie_agent.agent.notebooklm_client import NotebookLMClient, NotebookLMError, NotebookLMCache, NLM_RECOVERY_NEEDED
from retie_agent.agent.gemini_client import GeminiFileSearchClient, GeminiError
from retie_agent.agent.table_render import render_telegram_table, render_table_image
from retie_agent.agent.deep_answer import (
    run_deep_answer,
    deep_tools_available,
    NO_EVIDENCE_PHRASE,
)
from retie_agent.agent.intent import (
    classify_intent_v3,                   # clasificador único del grafo (classifier_node)
    extract_entities,                     # ref de tabla para el lookup canónico
    _TABLE_RE, _FULL_RE, _SMALLTALK_RE,   # re-exportados para compatibilidad
    _wants_table, _wants_full,
    _collapse_elongations,                # "graciaaas" → "gracias" (solo para regex)
)
from retie_agent.services.table_assets import CanonicalTable, lookup_table

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
    gemini_docs: Optional[str]            # respuesta de Gemini File Search (SPIKE Fase 1a)
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
    run_chromadb: bool         # rama Chroma incluida en el fan-out (según SECONDARY_RAG_SOURCE)
    run_notebooklm: bool       # rama NLM incluida en el fan-out (según SECONDARY_RAG_SOURCE)
    run_gemini: bool           # rama Gemini incluida en el fan-out (SPIKE Fase 1a)
    secondary_primary: str     # "notebooklm" | "gemini": cuál fuente secundaria alimenta answer_node
    search_query: Optional[str]  # standalone query (condensed from history) used for retrieval
    suggestions: List[str]  # follow-up questions offered to the user after the answer
    intent: Optional[str]        # taxonomía v3 (v1: exhaustiva | tabla | puntual | smalltalk)
    intent_source: Optional[str]  # "regex" | "llm" | "llm+escalated" (+"+guard")
    intent_meta: Optional[Dict[str, Any]]  # IntentResultV3.to_state_meta(): response_format,
                                           # output_length, complexity, needs_calculation,
                                           # confidence, entities, requires_rag
    source: Optional[str]        # "text" | "image" | "voice" | "suggestion" — origen del mensaje


# La clasificación de intención vive en retie_agent/agent/intent.py
# (classify_intent_v3: structured output + cascada + regex fallback; los regexes
# _TABLE_RE, _FULL_RE, _SMALLTALK_RE, _wants_table, _wants_full se re-exportan
# arriba por compatibilidad). classifier_node es quien la invoca.

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
    # Alargamientos ("graciaaas") se normalizan solo para elegir la respuesta:
    # sin esto, un agradecimiento alargado recibía el saludo de bienvenida.
    is_thanks = _SMALLTALK_THANKS_RE.search(_collapse_elongations(q))
    answer = _SMALLTALK_THANKS if is_thanks else _SMALLTALK_GREETING
    with span_ctx(None, "smalltalk_node", as_type="chain", span_input={"question": q}):
        return {"answer": answer}


# Respuesta para consultas vagas/incompletas ("que", "que es"): se pide precisión
# en vez de devolver el saludo de bienvenida o un genérico "sin evidencia".
_AMBIGUOUS_CLARIFY = (
    "🤔 ¿Podrías darme un poco más de detalle? Cuéntame sobre qué tema del RETIE o "
    "la NTC 2050 quieres saber —por ejemplo: puesta a tierra, calibres de conductor, "
    "tableros, tomas GFCI, distancias de seguridad— o escribe tu pregunta completa."
)


def _node_clarify(state: GraphState) -> GraphState:
    q = state.get("question", "")
    with span_ctx(None, "clarify_node", as_type="chain", span_input={"question": q}):
        return {"answer": _AMBIGUOUS_CLARIFY}


# Consulta fuera del dominio RETIE/NTC 2050 (clasificada por el LLM con confianza
# alta): cortesía fija SIN pagar el pipeline completo (query enrichment + 3
# retrievals + answer + enrich + suggest). Solo se llega aquí pasando el triple
# guardarraíl de intent.classify_intent_v3 (vía LLM + confianza ≥ INTENT_OOD_MIN_CONF).
_OUT_OF_DOMAIN_MSG = (
    "🙋 Soy un asistente especializado en normativa eléctrica colombiana "
    "(RETIE y NTC 2050), así que sobre ese tema no puedo ayudarte.\n\n"
    "Pregúntame, por ejemplo:\n"
    "• ¿Qué exige el RETIE sobre puesta a tierra?\n"
    "• Dame la tabla 220.55 de factores de demanda\n"
    "• ¿Es obligatorio el GFCI en baños?"
)


def _node_out_of_domain(state: GraphState) -> GraphState:
    q = state.get("question", "")
    with span_ctx(None, "out_of_domain_node", as_type="chain", span_input={"question": q}):
        return {"answer": _OUT_OF_DOMAIN_MSG}


def _node_image_answer(state: GraphState) -> GraphState:
    """Respuesta directa desde el contenido de una imagen (route == "image_direct").

    El clasificador marcó fuera_de_dominio, pero la consulta proviene de una
    imagen: lo leído por OCR/Vision viaja en la propia pregunta y suele contener
    la respuesta (p. ej. la clase de una etiqueta de eficiencia energética).
    Responde con el prompt de media en una sola llamada, sin RAG: ayuda con lo
    visible y aclara en una frase si el tema pertenece a otro reglamento
    (RETIQ, etc.). Si el LLM falla, cae al mensaje fijo de fuera de dominio
    (el comportamiento previo a esta ruta).
    """
    q = state.get("question", "")
    model = _resolve_model(state.get("agent_key"), explicit=None)
    history = state.get("history") or []
    with span_ctx(
        None, "image_answer_node", as_type="chain",
        span_input={"question": q, "model": model},
    ) as span:
        try:
            answer = _synthesize_simple(q, [], "", history, model, False, False, media_only=True)
        except Exception:
            answer = ""
        status = "ok" if (answer or "").strip() else "fallback_out_of_domain"
        if not (answer or "").strip():
            answer = _OUT_OF_DOMAIN_MSG
        if span is not None:
            try:
                span.update(output={"answer": answer, "status": status})
            except Exception:
                pass
        return {"answer": answer}


def make_state(question: str, *, user_id: str = "anon", session: str = "default", agent_key: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, source: Optional[str] = None) -> GraphState:
    return {
        "question": (question or "").strip(),
        "user_id": user_id or "anon",
        "session": session or "default",
        "agent_key": agent_key,
        "history": history or [],
        "source": source,
    }

_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))

ProgressCallback = Callable[[Dict[str, Any]], None]


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    message: str,
    **detail: Any,
) -> None:
    """Emite progreso estructurado sin acoplar el grafo a Telegram/Web/etc."""
    if progress_callback is None:
        return
    try:
        progress_callback({"stage": stage, "message": message, "detail": detail})
    except Exception:
        pass


def _progress_from_config(config: Optional[Dict[str, Any]]) -> Optional[ProgressCallback]:
    configurable = (config or {}).get("configurable") or {}
    cb = configurable.get("progress_callback")
    return cb if callable(cb) else None

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
def _node_query_enrichment(state: GraphState) -> GraphState:
    """Enriquece/desambigua la consulta ANTES del fan-out a las fuentes RAG.

    Dos disparadores:
      - Con historial: reescribe la pregunta de seguimiento como autocontenida
        ("¿y eso aplica en baja tensión?" no recupera nada útil sin el contexto
        previo) y expande siglas del dominio (SPT, DPS, GFCI…).
      - Sin historial: solo si el classifier marcó complexity=high (mensaje
        compuesto/ambiguo que se beneficia de la expansión).
    En el resto de casos es un pass-through sin costo. La pregunta original se
    conserva para el prompt final; aquí solo se produce `search_query`.

    (Ex condense_node — renombrado en Fase 2 del plan v3; la responsabilidad
    post-answer de pulir la redacción vive en enrich_node, que es otra cosa.)
    """
    q = state["question"]
    history = state.get("history") or []
    meta = state.get("intent_meta") or {}
    enabled = as_bool(getattr(settings, "QUERY_REWRITE_ENABLED", "true"))
    # Sin historial, la reescritura solo aporta cuando el mensaje es complejo.
    force = meta.get("complexity") == "high"
    if not enabled or not (history or force):
        return {"search_query": q}

    with span_ctx(
        None, "query_enrichment_node", as_type="chain",
        span_input={
            "question": q,
            "history_messages": len(history),
            "forced_by_complexity": force and not history,
        },
    ) as span:
        search_query = q
        try:
            prompt = QUERY_ENRICHMENT_PROMPT.format(
                history=format_history_for_condense(history), question=q
            )
            resp = create_chat_completion(
            _client,
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
            logger.warning("query_enrichment_node failed, using original question: %s", exc)

        if span is not None:
            try:
                span.update(output={
                    "search_query": search_query,
                    "rewritten": search_query != q,
                })
            except Exception:
                pass
        return {"search_query": search_query}


# Alias legacy: scripts/imports externos que referencien el nombre anterior.
# Retirar en el commit de limpieza de la Fase 5.
_node_condense = _node_query_enrichment


def _node_chromadb(state: GraphState, config=None) -> GraphState:
    """Rama de recuperación contra ChromaDB (paralela a notebooklm_node).

    Nodo independiente: escribe su salida en `chromadb_docs` para que, en la
    traza de Langfuse, al hacer clic en chromadb_node se vean exactamente los
    documentos/scores recuperados de Chroma — separados de lo que aportó NLM.
    """
    _emit_progress(
        _progress_from_config(config),
        "chromadb_node",
        "📚 Buscando en la base normativa…",
    )
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

def _resolve_rag_sources() -> tuple:
    """Decide qué fuente(s) de recuperación corren y cuál secundaria alimenta answer_node.

    Lee SECONDARY_RAG_SOURCE junto con los flags de disponibilidad de cada fuente.
    Devuelve (run_chroma, run_nlm, run_gemini, primary). Feature flag por entorno
    en Railway (se alterna sin redeploy):
      - notebooklm      → Chroma + NLM (default, retrocompatible)
      - gemini          → Chroma + Gemini; Gemini alimenta la respuesta
      - shadow          → los 3 corren; NLM alimenta (prod-safe), Gemini se
                          loguea para comparar calidad a igualdad de pregunta
      - chroma          → solo Chroma (alias: chromadb, solo-chroma, none)
      - solo-notebooklm → solo NLM, sin Chroma (alias: notebooklm-only)
      - solo-gemini     → solo Gemini, sin Chroma (alias: gemini-only)
    NLM/Gemini solo corren si además están disponibles (NOTEBOOKLM_ENABLED /
    key+store). Un valor desconocido cae al default retrocompatible.
    """
    raw = str(getattr(settings, "SECONDARY_RAG_SOURCE", "notebooklm"))
    # Railway guarda comillas literales si se pegan en el valor del dashboard;
    # se toleran aquí para que `"shadow"` no caiga silenciosamente al default.
    src = raw.strip().strip("\"'").strip().lower()
    nlm_on = str(getattr(settings, "NOTEBOOKLM_ENABLED", "false")).lower() in ("1", "true", "yes")
    gem_on = bool((getattr(settings, "GEMINI_API_KEY", None) or "").strip()
                  and (getattr(settings, "GEMINI_FILE_SEARCH_STORE", None) or "").strip())

    if src in ("chroma", "chromadb", "solo-chroma", "none"):
        return True, False, False, "notebooklm"
    if src == "gemini":
        return True, False, gem_on, "gemini"
    if src in ("solo-gemini", "gemini-only"):
        return False, False, gem_on, "gemini"
    if src in ("solo-notebooklm", "notebooklm-only"):
        return False, nlm_on, False, "notebooklm"
    if src in ("shadow", "all", "todos"):
        # Los 3; NLM sigue siendo el primario para no arriesgar producción.
        return True, nlm_on, gem_on, "notebooklm"
    # "notebooklm" (default) o valor desconocido → comportamiento actual.
    return True, nlm_on, False, "notebooklm"


def _node_classifier(state: GraphState, config=None) -> GraphState:
    """Entry point del grafo: clasifica la intención y publica IntentResultV3.

    Solo decide QUÉ quiere el usuario (intención + parámetros de entrega); la
    resolución de fuentes RAG vive en route_entry. El clasificador v3 es el
    único del grafo (structured output + cascada de modelos + fallback regex —
    ver intent.classify_intent_v3). `is_media`: el contenido derivado de
    imagen/voz nunca es smalltalk.
    """
    q = state.get("question", "")
    is_media = state.get("source") in ("image", "voice")
    channel = (state.get("metadata") or {}).get("channel", "telegram")
    _emit_progress(
        _progress_from_config(config),
        "classifier_node",
        "🧭 Clasificando tu consulta…",
    )
    res = classify_intent_v3(
        q, history=state.get("history"), is_media=is_media, channel=channel,
        media_source=state.get("source"),
    )
    with span_ctx(
        None, "classifier_node", as_type="chain",
        span_input={"question": q, "source": state.get("source"), "channel": channel},
    ) as span:
        if span is not None:
            try:
                span.update(output={
                    "intent": res.intent,
                    "intent_source": res.source,
                    "route": res.route,
                    "confidence": res.confidence,
                    "requires_rag": res.requires_rag,
                    "response_format": res.response_format,
                    "output_length": res.output_length,
                    "complexity": res.complexity,
                    "needs_calculation": res.needs_calculation,
                    "entities": res.entities,
                    "wants_table": res.wants_table,
                    "wants_full": res.wants_full,
                })
            except Exception:
                pass
    return {
        "route": res.route,
        "intent": res.intent,
        "intent_source": res.source,
        "intent_meta": res.to_state_meta(),
        "wants_table": res.wants_table,
        "wants_full": res.wants_full,
    }


def _node_route_entry(state: GraphState) -> GraphState:
    """Resuelve CON QUÉ fuentes RAG se responde (SECONDARY_RAG_SOURCE + disponibilidad).

    La clasificación de intención ya ocurrió en classifier_node (entry point).
    Aquí solo se publican los flags que `_fanout_retrieval` usa para armar el
    fan-out desde query_enrichment_node. `notebooklm_enabled` se conserva por
    compatibilidad.
    """
    run_chroma, run_nlm, run_gemini, secondary_primary = _resolve_rag_sources()
    flags = {
        "run_chromadb": run_chroma,
        "run_notebooklm": run_nlm,
        "run_gemini": run_gemini,
        "secondary_primary": secondary_primary,
    }
    with span_ctx(
        None, "route_entry", as_type="chain", span_input=flags,
    ) as span:
        if span is not None:
            try:
                span.update(output=flags)
            except Exception:
                pass
    return {"notebooklm_enabled": run_nlm, **flags}


def _synthesize_simple(
    question: str,
    hits: List[Dict[str, Any]],
    nlm_answer: str,
    history: List[Dict[str, str]],
    model: str,
    wants_table: bool,
    wants_full: bool,
    *,
    media_only: bool = False,
) -> str:
    """Síntesis clásica de UNA llamada (el cuerpo del answer_node previo al deep
    agent, extraído literal). Es la RED DE SEGURIDAD del deep agent — no un flag:
    corre cuando el agente agota su presupuesto, falla o devuelve vacío, y también
    resuelve el camino media_only (imagen/voz sin evidencia, sin agente)."""
    if media_only:
        # El contenido de la imagen/voz ya está en `question` (lo compuso el router).
        sys = SYSTEM_PROMPT_MEDIA
        prompt = question
    else:
        sys = SYSTEM_PROMPT_RETIE
        prompt = make_hybrid_prompt(
            hits, nlm_answer, question, is_admin=False, wants_table=wants_table
        )
    messages = [{"role": "system", "content": sys}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})

    temperature = 0.0
    # Tablas y respuestas exhaustivas (todos los numerales a–y) necesitan mucho
    # más presupuesto de salida; las respuestas normales conservan el default.
    default_max = getattr(settings, "MAX_TOKENS", 600)
    max_tokens = 4000 if (wants_table or wants_full) else default_max

    resp = create_chat_completion(
        _client,
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
                "context_chunks": len(hits),
                "sources": [(h.get("meta", {}) or {}).get("source") for h in hits],
                "media_only": media_only,
            },
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        )
    except Exception:
        pass
    return answer


def _node_answer(state: GraphState, config=None) -> GraphState:
    """Orquestador del deep agent (Fase 4 del plan v3).

    Fan-in de las ramas de recuperación (cada una escribió su clave propia:
    chromadb_docs / notebooklm_docs / gemini_docs) y delegación en el deep agent
    (deep_answer.run_deep_answer), que puede re-consultar las fuentes con
    queries refinadas si la evidencia inicial no basta. Timeout / error /
    respuesta vacía degradan a _synthesize_simple (el camino clásico) sin
    exponer el fallo al usuario.

    CONTRATO INTOCABLE del nodo: {"answer", "hits", "nlm_answer", "route"} —
    table_node consume answer+hits+nlm_answer, stylist_node arma las citas desde
    hits y _after_answer rutea por route. `config` lo inyecta langgraph con el
    CallbackHandler de Langfuse; pasarlo al agente es imprescindible (corre en
    otro thread; sin él la traza del subgrafo se corta).
    """
    q = state["question"]
    agent_key = state.get("agent_key")
    hits = state.get("chromadb_docs") or []
    # Fuente secundaria que alimenta la respuesta: según SECONDARY_RAG_SOURCE, es
    # NotebookLM o Gemini File Search (en "shadow" ambos corrieron, pero solo el
    # `secondary_primary` entra al agente; el otro queda en su span). Se sigue
    # exponiendo como `nlm_answer` para no tocar el contrato de table/stylist.
    secondary_primary = state.get("secondary_primary", "notebooklm")
    if secondary_primary == "gemini":
        nlm_answer = (state.get("gemini_docs") or "").strip()
    else:
        nlm_answer = (state.get("notebooklm_docs") or "").strip()
    source = state.get("source")
    wants_table = state.get("wants_table", False)
    wants_full = state.get("wants_full", False)
    model = _resolve_model(agent_key, explicit=None)
    history = state.get("history") or []
    progress_callback = _progress_from_config(config)

    no_evidence = not hits and not nlm_answer

    # Imagen/voz sin evidencia recuperada: el material a interpretar (OCR/Vision
    # o transcripción) viaja en la propia pregunta → prompt dedicado, SIN agente.
    if no_evidence and source in ("image", "voice"):
        _emit_progress(
            progress_callback,
            "media_answer",
            "🖼️ Interpretando la imagen enviada…"
            if source == "image"
            else "🎙️ Interpretando la nota de voz…",
            source=source,
        )
        with span_ctx(
            None, "answer_node",
            {"model": model, "context_chunks": 0, "agent_key": agent_key or "default"},
            as_type="chain",
            span_input={"question": q, "media_only": True},
        ) as span:
            answer = _synthesize_simple(
                q, [], "", history, model, False, False, media_only=True
            )
            if span is not None:
                try:
                    span.update(output={"answer": answer, "status": "media_only"})
                except Exception:
                    pass
            return {"answer": answer, "hits": hits, "nlm_answer": nlm_answer, "route": "answer"}

    # Sin evidencia inicial Y sin ninguna tool de re-consulta → corte temprano
    # (como el grafo clásico). CAMBIO DELIBERADO respecto al grafo previo: sin
    # evidencia pero CON tools, el agente corre — puede reformular y encontrar.
    if no_evidence and not deep_tools_available():
        _emit_progress(
            progress_callback,
            "no_evidence",
            "🔎 Sin evidencia suficiente en las fuentes disponibles…",
        )
        return {
            "answer": NO_EVIDENCE_PHRASE,
            "hits": hits,
            "nlm_answer": nlm_answer,
            "route": "no_context",
        }

    with span_ctx(
        None, "answer_node",
        {"model": model, "context_chunks": len(hits), "agent_key": agent_key or "default"},
        as_type="chain",
        span_input={"question": q, "intent": state.get("intent")},
    ) as span:
        _emit_progress(
            progress_callback,
            "answer_node",
            "📖 Analizando la evidencia normativa recuperada…",
            hits=len(hits),
            has_secondary=bool(nlm_answer),
            intent=state.get("intent"),
        )
        result = run_deep_answer(
            q,
            initial_hits=hits,
            secondary_answer=nlm_answer,
            history=history,
            agent_key=agent_key,
            intent=state.get("intent"),
            intent_meta=state.get("intent_meta"),
            wants_table=wants_table,
            wants_full=wants_full,
            model=model,
            config=config,
            progress_callback=progress_callback,
        )
        answer = result.answer
        status = result.status

        if status != "ok" or not answer.strip():
            # Red de seguridad (no flag): el camino clásico de una sola llamada.
            if no_evidence:
                answer = NO_EVIDENCE_PHRASE
            else:
                try:
                    _emit_progress(
                        progress_callback,
                        "fallback_simple",
                        "✍️ Elaborando la respuesta con la evidencia disponible…",
                        status=status,
                    )
                    answer = _synthesize_simple(
                        q, hits, nlm_answer, history, model, wants_table, wants_full
                    )
                    status = f"{status}->simple"
                except Exception as exc:
                    logger.warning("answer_node: síntesis simple también falló: %s", exc)
                    answer = NO_EVIDENCE_PHRASE

        # hits = iniciales + acumulados por las tools del agente (deduplicados):
        # las citas del stylist y el contexto crudo del table_node salen de aquí.
        all_hits = _dedupe_hits(hits + result.hits) if result.hits else hits

        try:
            log_generation(
                None,
                name="deep_answer_agent",
                input_text=q,
                output_text=answer,
                model=model,
                usage=result.usage,
                metadata={
                    "agent_key": agent_key or "default",
                    "status": status,
                    "skill": result.skill,
                    "tool_calls": result.tool_calls,
                    "n_hits_inicial": len(hits),
                    "n_hits_final": len(all_hits),
                },
            )
        except Exception:
            pass

        if span is not None:
            try:
                span.update(output={
                    "answer": answer,
                    "status": status,
                    "skill": result.skill,
                    "tool_calls": result.tool_calls,
                    "n_hits_inicial": len(hits),
                    "n_hits_final": len(all_hits),
                })
            except Exception:
                pass

        route = "no_context" if answer.strip() == NO_EVIDENCE_PHRASE else "answer"
        return {"answer": answer, "hits": all_hits, "nlm_answer": nlm_answer, "route": route}

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


_gemini_client: Optional[GeminiFileSearchClient] = None


def _get_gemini_client() -> GeminiFileSearchClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiFileSearchClient(
            api_key=getattr(settings, "GEMINI_API_KEY", None),
            model=getattr(settings, "GEMINI_MODEL", "gemini-flash-latest"),
            store=getattr(settings, "GEMINI_FILE_SEARCH_STORE", None),
            timeout=float(getattr(settings, "GEMINI_TIMEOUT", 120.0)),
        )
    return _gemini_client


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


def _node_notebooklm(state: GraphState, config=None) -> GraphState:
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

    _emit_progress(
        _progress_from_config(config),
        "notebooklm_node",
        "📖 Consultando NotebookLM…",
    )
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
#  Gemini File Search Node (rama independiente — SPIKE Fase 1a)
# ===============================
def _node_gemini(state: GraphState, config=None) -> GraphState:
    """Rama de recuperación contra Gemini File Search (paralela a chromadb_node).

    Espeja a notebooklm_node: escribe su salida en `gemini_docs` y crea su propio
    span en Langfuse para poder comparar, a igualdad de pregunta, la respuesta de
    Gemini vs la de NotebookLM (validación de calidad del piloto A/B). Solo se
    ejecuta cuando el fan-out lo incluye (SECONDARY_RAG_SOURCE=gemini|shadow|
    solo-gemini y hay key + store). Degrada a "" ante cualquier fallo: nunca
    bloquea el fan-in.
    """
    q = state.get("search_query") or state["question"]
    timeout = float(getattr(settings, "GEMINI_TIMEOUT", 120.0))
    _emit_progress(
        _progress_from_config(config),
        "gemini_node",
        "🌐 Consultando el corpus vía Gemini…",
    )

    with span_ctx(
        None, "gemini_node", as_type="retriever",
        span_input={"question": q},
    ) as span:
        answer = ""
        sources: List[Dict[str, Any]] = []
        error_msg = ""
        status = "disabled"

        client = _get_gemini_client()
        if not client.is_available():
            status = "unconfigured"  # falta GEMINI_API_KEY o GEMINI_FILE_SEARCH_STORE
        else:
            fut = _POOL.submit(client.ask_question, q)
            try:
                raw_answer, raw_sources = fut.result(timeout=timeout)
                answer = (raw_answer or "").strip()
                sources = raw_sources or []
                status = "ok" if answer else "empty"
            except FutureTimeoutError:
                status = "timeout"
                error_msg = f"Gemini no respondió en {timeout:.0f}s"
                logger.warning("gemini_node: espera de %.0fs agotada", timeout)
            except Exception as exc:
                status = "error"
                error_msg = str(exc)
                logger.warning("gemini_node query failed: %s", exc)

        if span is not None:
            try:
                output: Dict[str, Any] = {
                    "answer": answer,
                    "answer_len": len(answer),
                    "answer_preview": answer[:1000],
                    "sources": _format_nlm_sources(sources),
                    "sources_count": len(sources),
                    "status": status,
                }
                if error_msg:
                    output["error"] = error_msg[:800]
                span.update(
                    output=output,
                    metadata={
                        "status": status,
                        "model": getattr(settings, "GEMINI_MODEL", None),
                        "store": getattr(settings, "GEMINI_FILE_SEARCH_STORE", None),
                        "wait_budget_s": timeout,
                        "question": q,
                    },
                )
            except Exception:
                pass

        return {"gemini_docs": answer}


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
    repaired = repaired.rstrip()
    while repaired and repaired[-1] in ",":
        repaired = repaired[:-1].rstrip()
    for opener in reversed(stack):
        repaired += "]" if opener == "[" else "}"
    return repaired


_NUM_RE = re.compile(r"\b\d[\d.,]*\b")


def _count_raw_row_candidates(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(_NUM_RE.findall(stripped)) >= 2:
            count += 1
    return count


def _parse_table_json(raw: str) -> tuple:
    """Extrae y valida el JSON {headers, rows} de la salida del LLM.

    Tolera fences markdown, texto alrededor y truncamiento (token limit).
    Devuelve (table_dict_o_None, was_repaired)."""
    if not raw:
        return None, False
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    start = text.find("{")
    if start == -1:
        return None, False
    candidate = text[start:]
    end = candidate.rfind("}")
    snippet = candidate[: end + 1] if end != -1 else candidate

    data = None
    was_repaired = False
    for i, attempt in enumerate((snippet, _close_truncated_json(candidate))):
        try:
            data = _json.loads(attempt)
            was_repaired = i == 1
            break
        except _json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return None, False

    headers = data.get("headers")
    rows = data.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list):
        return None, False
    if not headers or not rows:
        return None, False

    ncol = len(headers)
    norm_rows: List[List[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))
        norm_rows.append(row[:ncol])
    if not norm_rows:
        return None, False

    title = data.get("title")
    return {
        "title": title if isinstance(title, str) else None,
        "headers": headers,
        "rows": norm_rows,
    }, was_repaired


def _fetch_continuation_rows(
    partial_table: Dict[str, Any],
    info: str,
    model: str,
    max_tokens: int,
) -> tuple:
    """Pide al LLM las filas faltantes de una tabla cortada.

    Devuelve (filas_extra, continuación_también_cortada)."""
    headers = partial_table.get("headers", [])
    existing_rows = partial_table.get("rows", [])
    last_row = existing_rows[-1] if existing_rows else []
    prompt = (
        f"La extracción de la tabla fue cortada ({len(existing_rows)} filas obtenidas). "
        f"Encabezados: {_json.dumps(headers, ensure_ascii=False)}. "
        f"Última fila extraída: {_json.dumps(last_row, ensure_ascii=False)}.\n"
        f"Devuelve SOLO las filas faltantes como un array JSON de arrays. "
        f"Si no quedan filas, responde []. No repitas la última fila ya extraída.\n\n"
        f"INFORMACIÓN:\n{info}"
    )
    try:
        resp = create_chat_completion(
            _client,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "Eres un formateador de datos a tablas. Solo devuelves JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        cont_truncated = resp.choices[0].finish_reason == "length"
        cont_raw = (resp.choices[0].message.content or "").strip()
        if cont_raw.startswith("```"):
            cont_raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", cont_raw, flags=re.IGNORECASE).strip()
        start = cont_raw.find("[")
        if start == -1:
            return [], cont_truncated
        arr_text = cont_raw[start:]
        end = arr_text.rfind("]")
        if end != -1:
            arr_text = arr_text[: end + 1]
        elif cont_truncated:
            # Array sin cerrar por corte de tokens: se rescatan las filas
            # completas cerrando el JSON igual que en _parse_table_json.
            arr_text = _close_truncated_json(arr_text)
        parsed = _json.loads(arr_text)
        if not isinstance(parsed, list):
            return [], cont_truncated
        ncol = len(headers)
        result = []
        for row in parsed:
            if not isinstance(row, list):
                continue
            if len(row) < ncol:
                row = row + [""] * (ncol - len(row))
            result.append(row[:ncol])
        return result, cont_truncated
    except Exception as exc:
        logger.warning("table_node: continuación de filas falló: %s", exc)
        return [], False


_TABLE_PARTIAL_NOTE = (
    "⚠️ No pude reconstruir la tabla completa en este intento "
    "(la fuente NotebookLM no estuvo disponible). "
    "Esta es la información encontrada en la base local:\n\n"
)

_TABLE_INCOMPLETE_NOTE = (
    "⚠️ Esta tabla puede estar incompleta: el contexto fue truncado o la extracción "
    "se cortó antes de terminar. Puede haber filas faltantes.\n\n"
)

_TABLE_RAW_CONTEXT_CHARS = 24000


def _requested_table_ref(state: GraphState) -> Optional[str]:
    """Referencia de tabla pedida por el usuario: primero las entidades del
    classifier (intent_meta), si no, extracción regex directa de la pregunta."""
    meta = state.get("intent_meta") or {}
    for e in meta.get("entities") or []:
        if isinstance(e, dict) and e.get("type") == "tabla" and e.get("value"):
            return str(e["value"])
    try:
        for e in extract_entities(state.get("question", "")):
            if e.get("type") == "tabla" and e.get("value"):
                return e["value"]
    except Exception:
        pass
    return None


def _serve_canonical_table(canonical: CanonicalTable, state: GraphState) -> Optional[GraphState]:
    """Entrega la tabla desde el activo canónico del índice v2 (cero LLM).

    - estado verificada/extraida con JSON → render PNG desde headers/rows
      canónicos (cada valor en su columna, cero filas faltantes por diseño).
    - estado solo_imagen → el PNG original del PDF (fidelidad 100 %).
    - revision / sin datos → None: el llamador cae al camino LLM.
    Los footnotes y la línea "Fuente" (notes del JSON) van en el caption.
    """
    question = state.get("question", "")
    title = canonical.titulo or f"Tabla {canonical.tabla_id}"

    # Entrada de un índice previo al fix de IDs truncados (la referencia es más
    # específica que el id registrado): su JSON puede ser OTRA tabla (las
    # variantes .a/.b se fusionaban bajo un id). No confiable → camino LLM.
    if canonical.match == "ref_extends_stored":
        return None

    image: Optional[bytes] = None
    final_text: Optional[str] = None
    status: Optional[str] = None

    if canonical.estado in ("verificada", "extraida") and canonical.headers and canonical.rows:
        caption = f"📊 Tabla {canonical.tabla_id}"
        clean_title = title.strip()
        if clean_title and not clean_title.lower().startswith(f"tabla {canonical.tabla_id}".lower()):
            caption += f" — {clean_title}"
        try:
            image = render_table_image(canonical.headers, canonical.rows, title=title)
            final_text = caption
            status = "canonical_json"
        except Exception as img_exc:
            logger.warning("tabla canónica %s: render de imagen falló, tabla de texto: %s",
                           canonical.tabla_id, img_exc)
            final_text = render_telegram_table(canonical.headers, canonical.rows)
            status = "canonical_text"
        if canonical.notes:
            final_text += "\n\n" + canonical.notes.strip()[:400]
    elif canonical.estado == "solo_imagen" and canonical.png_path is not None:
        try:
            image = canonical.png_path.read_bytes()
            final_text = f"📊 Tabla {canonical.tabla_id} (imagen original del documento)"
            status = "canonical_png"
        except Exception:
            return None
    else:
        return None

    with span_ctx(
        None, "table_node", {"tabla_id": canonical.tabla_id},
        as_type="chain",
        span_input={"question": question, "tabla_id": canonical.tabla_id},
    ) as span:
        if span is not None:
            try:
                span.update(
                    output={"preview": final_text[:200], "has_image": image is not None},
                    metadata={
                        "status": status,
                        "doc_id": canonical.doc_id,
                        "estado_extraccion": canonical.estado,
                        "n_filas": len(canonical.rows),
                        "n_cols": len(canonical.headers),
                    },
                )
            except Exception:
                pass
        try:
            log_generation(
                None, name="table_node", input_text=question, output_text=final_text,
                model="canonical_asset",
                metadata={"status": status, "tabla_id": canonical.tabla_id,
                          "has_image": image is not None},
            )
        except Exception:
            pass

    out: GraphState = {"answer": final_text, "route": "stylist_node"}
    if image is not None:
        out["table_image"] = image
    return out


def _node_table(state: GraphState) -> GraphState:
    base_answer = state.get("answer", "") or ""
    question = state.get("question", "")

    # ── Camino canónico (índice v2): la tabla se sirve como unidad atómica ────
    # desde el registro de activos (JSON estructurado en la indexación o PNG
    # original), en vez de que un LLM la RECONSTRUYA adivinando desde chunks de
    # texto plano — que es como se perdían filas y se corrían columnas.
    ref = _requested_table_ref(state)
    if ref:
        try:
            canonical = lookup_table(ref)
        except Exception:
            canonical = None
        if canonical is not None:
            served = _serve_canonical_table(canonical, state)
            if served is not None:
                return served

    if not base_answer.strip():
        return {"answer": base_answer, "route": "stylist_node"}

    _hits_sorted = sorted(
        (h for h in (state.get("hits") or []) if h.get("text")),
        key=lambda h: (
            str((h.get("meta") or {}).get("source", "")),
            (h.get("meta") or {}).get("page", 0) or 0,
        ),
    )

    ctx_limit = int(getattr(settings, "TABLE_RAW_CONTEXT_CHARS", _TABLE_RAW_CONTEXT_CHARS))
    raw_full = "\n\n".join(h.get("text", "") for h in _hits_sorted)
    context_truncated = len(raw_full) > ctx_limit
    if context_truncated:
        logger.warning(
            "table_node: contexto crudo truncado de %d a %d chars — puede haber filas faltantes",
            len(raw_full), ctx_limit,
        )
    raw_chunks = raw_full[:ctx_limit]
    raw_row_candidates = _count_raw_row_candidates(raw_chunks)

    info = base_answer
    if raw_chunks:
        info = f"{base_answer}\n\nFRAGMENTOS LITERALES DEL DOCUMENTO:\n{raw_chunks}"

    model = _resolve_model(state.get("agent_key"), explicit=None)
    prompt = _TABLE_PROMPT.format(question=question, answer=info)

    table: Optional[Dict[str, Any]] = None
    sanity_ok: Optional[bool] = None
    incomplete = context_truncated

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
            resp = create_chat_completion(
                _client,
                model=model,
                temperature=0.0,
                max_tokens=_TABLE_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "Eres un formateador de datos a tablas. Solo devuelves JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            finish_reason = resp.choices[0].finish_reason

            table, was_repaired = _parse_table_json(raw)

            if finish_reason == "length" or was_repaired:
                logger.warning(
                    "table_node: JSON cortado (finish_reason=%r, reparado=%s) — solicitando continuación",
                    finish_reason, was_repaired,
                )
                incomplete = True
                if table:
                    extra_rows, cont_truncated = _fetch_continuation_rows(
                        table, info, model, _TABLE_MAX_TOKENS
                    )
                    if extra_rows:
                        table["rows"].extend(extra_rows)
                        logger.info("table_node: continuación añadió %d filas", len(extra_rows))
                        # Completa solo si la continuación tampoco se cortó.
                        incomplete = cont_truncated

            if table and raw_row_candidates > 5:
                sanity_ok = len(table["rows"]) >= raw_row_candidates * 0.5
                if not sanity_ok:
                    logger.warning(
                        "table_node: sanity check — %d filas extraídas vs %d candidatas en contexto",
                        len(table["rows"]), raw_row_candidates,
                    )
                    incomplete = True

            if not table:
                logger.warning("table_node: no se pudo parsear JSON (%d chars): %s",
                               len(raw), raw[:300])
            if table:
                title = table.get("title")
                try:
                    image = render_table_image(table["headers"], table["rows"], title=title)
                    caption = f"📊 {title}" if title else "📊 Tabla solicitada"
                    final_text = (_TABLE_INCOMPLETE_NOTE + caption) if incomplete else caption
                except Exception as img_exc:
                    status = "text_fallback"
                    logger.warning("table image failed, using text table: %s", img_exc)
                    body = render_telegram_table(table["headers"], table["rows"])
                    final_text = (_TABLE_INCOMPLETE_NOTE + body) if incomplete else body
            else:
                status = "not_tabular"
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
                    metadata={
                        "status": status,
                        "context_truncated": context_truncated,
                        "raw_row_candidates": raw_row_candidates,
                        "sanity_ok": sanity_ok,
                    },
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


def _node_enrich(state: GraphState, config=None) -> GraphState:
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
    _emit_progress(
        _progress_from_config(config),
        "enrich_node",
        "🪶 Puliendo la redacción…",
    )

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
    "genera {n} preguntas de seguimiento MUY CORTAS (máx. 6 palabras y 40 "
    "caracteres cada una — van como texto de un botón de Telegram, no caben "
    "preguntas largas) que el usuario probablemente quiera hacer a continuación "
    "sobre el mismo tema.\n"
    "Reglas:\n"
    "- Directas al grano, sin rodeos ni conectores de más.\n"
    "- Preguntas concretas y respondibles con el RETIE o la NTC 2050.\n"
    "- No repitas preguntas que el usuario ya hizo.\n"
    "- Responde ÚNICAMENTE con un array JSON de strings, sin texto adicional.\n"
    'Ejemplo: ["¿Calibre mínimo para 40 A?", "¿Resistencia máxima permitida?"]\n\n'
    "ÚLTIMOS MENSAJES DEL USUARIO:\n{user_messages}\n\n"
    "RESPUESTA DADA (resumen):\n{answer}\n"
)


def _parse_suggestions(raw: str, limit: int) -> List[str]:
    """Extrae el array JSON de la salida del LLM (tolerante a fences/texto/truncamiento).

    Mismo patrón de reparación que _parse_table_json: si el array se cortó a
    mitad de una pregunta por el límite de tokens, _close_truncated_json lo
    balancea y se descarta el último elemento (puede venir incompleto) en vez
    de perder TODAS las sugerencias por un solo ítem malformado — la causa más
    probable de que a veces solo aparezca 1 de las 3 sugerencias esperadas.
    """
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    start = text.find("[")
    if start == -1:
        return []
    candidate = text[start:]
    end = candidate.rfind("]")
    snippet = candidate[: end + 1] if end != -1 else candidate

    data = None
    was_repaired = False
    for i, attempt in enumerate((snippet, _close_truncated_json(candidate))):
        try:
            data = _json.loads(attempt)
            was_repaired = i == 1
            break
        except _json.JSONDecodeError:
            continue
    if not isinstance(data, list):
        return []

    out = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    if was_repaired and out:
        out = out[:-1]  # el último elemento del array reparado puede venir cortado
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
            resp = create_chat_completion(
            _client,
                model=_resolve_model(state.get("agent_key"), explicit=None),
                temperature=0.7,
                # 200 se quedaba corto para n=3 preguntas de hasta 12 palabras +
                # sintaxis JSON: el array llegaba a cortarse a mitad de la última
                # pregunta y _parse_suggestions perdía sugerencias de más.
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
            )
            suggestions = _parse_suggestions(resp.choices[0].message.content or "", n)
            if len(suggestions) < n:
                logger.warning(
                    "suggest_node: se pidieron %d sugerencias, se obtuvieron %d (raw=%r)",
                    n, len(suggestions), (resp.choices[0].message.content or "")[:300],
                )
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


_HTML_PLACEHOLDER = "\x00HTML{}\x00"
_TELEGRAM_PRE_RE = re.compile(r"<pre\b[^>]*>.*?</pre>", re.IGNORECASE | re.DOTALL)
_MARKDOWN_FENCE_RE = re.compile(r"```(?:[^\n`]*)\n?(.*?)```", re.DOTALL)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
# El modelo (o una fuente secundaria como Gemini/NotebookLM) a veces escribe
# HTML crudo en vez de **negrilla** markdown. Estas son las etiquetas simples
# que Telegram soporta sin atributos — se preservan tal cual llegan en vez de
# perder el énfasis que el modelo quiso dar.
_SAFE_INLINE_TAG_RE = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|code)>", re.IGNORECASE
)
# Cualquier otra etiqueta ("<" + letra inicial, lo que excluye comparaciones
# numéricas tipo "<600V" o "< 5") se descarta — nunca se deja llegar a
# html.escape(), que la mostraría como &lt;b&gt; visible.
_RAW_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^<>]*>")


def _protect_html_block(blocks: List[str], value: str) -> str:
    token = _HTML_PLACEHOLDER.format(len(blocks))
    blocks.append(value)
    return token


def _restore_html_blocks(text: str, blocks: List[str]) -> str:
    for i, block in enumerate(blocks):
        text = text.replace(_HTML_PLACEHOLDER.format(i), block)
    return text


def _format_markdownish_as_telegram_html(text: str) -> str:
    """Convierte una respuesta interna tipo Markdown a HTML seguro de Telegram.

    El resto del sistema usa `response_format=markdown` como formato lógico, pero
    el bot de Telegram está en `parse_mode=HTML`. Esta función es deliberadamente
    conservadora: preserva bloques `<pre>` ya renderizados por `table_node`,
    convierte solo Markdown común y escapa todo lo demás.
    """
    raw = str(text or "")
    if not raw:
        return ""

    blocks: List[str] = []

    # 1) Preservar tablas ya generadas como HTML seguro (<pre>...</pre>).
    raw = _TELEGRAM_PRE_RE.sub(
        lambda m: _protect_html_block(blocks, m.group(0)),
        raw,
    )

    # 1.5) HTML crudo colado por el modelo/fuente secundaria: las etiquetas
    # simples de Telegram se protegen tal cual (conservan el énfasis), el
    # resto se descarta. Sin este paso, html.escape() (4) las muestra como
    # &lt;b&gt; visible — el bug reportado en producción.
    raw = _SAFE_INLINE_TAG_RE.sub(
        lambda m: _protect_html_block(blocks, m.group(0)),
        raw,
    )
    raw = _RAW_HTML_TAG_RE.sub("", raw)

    # 2) Convertir fenced code Markdown en <pre> escapado.
    raw = _MARKDOWN_FENCE_RE.sub(
        lambda m: _protect_html_block(blocks, f"<pre>{html.escape(m.group(1).strip())}</pre>"),
        raw,
    )

    # 3) Convertir enlaces seguros a <a>; el texto y href van escapados.
    def _link_repl(match: re.Match[str]) -> str:
        label = html.escape(match.group(1).strip())
        href = html.escape(match.group(2).strip(), quote=True)
        return _protect_html_block(blocks, f'<a href="{href}">{label}</a>')

    raw = _MARKDOWN_LINK_RE.sub(_link_repl, raw)

    # 4) Escapar todo el texto libre antes de reinsertar markup propio.
    escaped = html.escape(raw)

    # 5) Títulos Markdown → negrilla Telegram. No hay <h1>/<h2> en Telegram.
    escaped = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$",
        lambda m: f"<b>{m.group(1).strip()}</b>",
        escaped,
    )

    # 6) Énfasis Markdown básico. Se evita tocar saltos de línea para no comerse
    # listas/tablas; Telegram acepta <b>/<i>/<code>.
    escaped = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", escaped)

    # 7) Bullets Markdown → bullets más naturales en clientes móviles.
    escaped = re.sub(r"(?m)^(\s*)[-+]\s+", r"\1• ", escaped)

    return _restore_html_blocks(escaped, blocks)


def _format_as_whatsapp_text(text: str) -> str:
    """Salida textual portable para WhatsApp.

    WhatsApp usa una variante simple de Markdown. Mantener texto casi plano evita
    llevar HTML de Telegram a otros canales; los títulos Markdown siguen siendo
    legibles y las tablas <pre> se degradan a texto.
    """
    raw = str(text or "")
    raw = re.sub(r"</?pre\b[^>]*>", "```", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(raw).strip()


def _format_as_web_html(text: str) -> str:
    """HTML básico y seguro para superficies web futuras."""
    telegram_html = _format_markdownish_as_telegram_html(text)
    paragraphs = [
        p.replace("\n", "<br>")
        for p in re.split(r"\n{2,}", telegram_html)
        if p.strip()
    ]
    return "\n".join(f"<p>{p}</p>" for p in paragraphs)


def _format_for_channel(text: str, channel: str) -> tuple[str, str]:
    """Devuelve `(texto_formateado, formato)` según el canal de entrega."""
    normalized = (channel or "telegram").lower()
    if normalized == "telegram":
        return _format_markdownish_as_telegram_html(text), "telegram_html"
    if normalized == "whatsapp":
        return _format_as_whatsapp_text(text), "whatsapp_text"
    if normalized == "web":
        return _format_as_web_html(text), "web_html"
    return str(text or ""), "plain_text"


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
        span_input={
            "question": question,
            "channel": channel,
            "answer_preview": str(enriched_answer or "")[:2000],
            "sources_count": len(hits),
            "has_image": table_image is not None,
        },
    ) as span:
        try:
            styled_text, output_format = _format_for_channel(enriched_answer, channel)
            styled_sources_text, _ = _format_for_channel(sources_text, channel)

            # Unified JSON payload (future-proof)
            styled_payload = {
                "channel": channel,
                "format": output_format,
                "user_message": question,
                "formatted_response": styled_text,
                "sources": sources,
                "sources_text": styled_sources_text,
                "suggestions": state.get("suggestions") or [],
                "image": table_image,  # PNG bytes → router sends it as a photo
                "timestamp": _utcnow_iso(),
            }

            status = "ok"

        except Exception as e:
            styled_payload = {
                "channel": channel,
                "format": "plain_text",
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

        # Dejar el resultado visible en el nodo del árbol de Langfuse. Sin esto,
        # el span existe pero la UI muestra Output=undefined; `log_generation`
        # registra otra observación hija, no el output del nodo seleccionado.
        if span is not None:
            try:
                span_output = {k: v for k, v in styled_payload.items() if k != "image"}
                span_output["image"] = f"<{len(table_image)} bytes PNG>" if table_image else None
                span_output["status"] = status
                span.update(output=span_output)
            except Exception:
                pass

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
    g.add_node("classifier_node", _node_classifier)
    g.add_node("route_entry", _node_route_entry)
    g.add_node("out_of_domain_node", _node_out_of_domain)
    g.add_node("image_answer_node", _node_image_answer)
    g.add_node("query_enrichment_node", _node_query_enrichment)
    # Ramas de recuperación INDEPENDIENTES → cada una genera su propio span en
    # Langfuse (chromadb_node vs notebooklm_node), con su salida en clave propia.
    g.add_node("chromadb_node", _node_chromadb)
    g.add_node("notebooklm_node", _node_notebooklm)
    g.add_node("gemini_node", _node_gemini)
    g.add_node("answer_node", _node_answer)
    g.add_node("enrich_node", _node_enrich)
    g.add_node("stylist_node", _node_stylist)
    g.add_node("no_context", _node_no_context)
    g.add_node("table_node", _node_table)
    g.add_node("smalltalk_node", _node_smalltalk)
    g.add_node("clarify_node", _node_clarify)
    g.add_node("suggest_node", _node_suggest)

    # Entry: classifier_node decide QUÉ quiere el usuario. Los short-circuits
    # (smalltalk / ambiguous / fuera de dominio) responden con mensaje fijo sin
    # pagar RAG; image_direct responde desde el contenido leído de la imagen
    # (una llamada LLM, sin RAG); el resto pasa por route_entry (fuentes) →
    # query_enrichment_node (consulta autocontenida) → fan-out de recuperación.
    g.set_entry_point("classifier_node")

    def _after_classifier(s: GraphState) -> str:
        route = s.get("route")
        if route == "smalltalk":
            return "smalltalk"
        if route == "ambiguous":
            return "ambiguous"
        if route == "out_of_domain":
            return "out_of_domain"
        if route == "image_direct":
            return "image_direct"
        return "route_entry"

    g.add_conditional_edges(
        "classifier_node",
        _after_classifier,
        {
            "smalltalk": "smalltalk_node",
            "ambiguous": "clarify_node",
            "out_of_domain": "out_of_domain_node",
            "image_direct": "image_answer_node",
            "route_entry": "route_entry",
        },
    )
    g.add_edge("smalltalk_node", "stylist_node")
    g.add_edge("clarify_node", "stylist_node")
    g.add_edge("out_of_domain_node", "stylist_node")
    g.add_edge("image_answer_node", "stylist_node")
    g.add_edge("route_entry", "query_enrichment_node")

    # ── Fan-out de recuperación (ramas paralelas que convergen) ──────────────
    # La conditional edge devuelve la LISTA de ramas a ejecutar según los flags
    # publicados por route_entry (SECONDARY_RAG_SOURCE):
    #   - notebooklm      → ["chromadb_node", "notebooklm_node"]
    #   - gemini          → ["chromadb_node", "gemini_node"]
    #   - shadow          → ["chromadb_node", "notebooklm_node", "gemini_node"]  (A/B)
    #   - chroma          → ["chromadb_node"]
    #   - solo-notebooklm → ["notebooklm_node"]
    #   - solo-gemini     → ["gemini_node"]
    # LangGraph corre todas en el mismo super-step y espera a que terminen
    # (fan-in) antes de ejecutar answer_node. Las ramas no incluidas ni aparecen
    # en la traza.
    def _fanout_retrieval(s: GraphState) -> List[str]:
        targets = []
        if s.get("run_chromadb", True):
            targets.append("chromadb_node")
        if s.get("run_notebooklm"):
            targets.append("notebooklm_node")
        if s.get("run_gemini"):
            targets.append("gemini_node")
        # Nunca vacío: si la única fuente pedida no está disponible (p. ej.
        # solo-gemini sin key/store), se degrada a Chroma en vez de romper el grafo.
        return targets or ["chromadb_node"]

    g.add_conditional_edges(
        "query_enrichment_node",
        _fanout_retrieval,
        {
            "chromadb_node": "chromadb_node",
            "notebooklm_node": "notebooklm_node",
            "gemini_node": "gemini_node",
        },
    )

    # Fan-in: las ramas activas convergen en answer_node. LangGraph lo ejecuta una
    # sola vez, tras completarse las ramas que se hayan activado en el fan-out.
    g.add_edge("chromadb_node", "answer_node")
    g.add_edge("notebooklm_node", "answer_node")
    g.add_edge("gemini_node", "answer_node")

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
    g.add_edge("no_context", "stylist_node")
    g.add_edge("stylist_node", END)

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
    progress_callback: Optional[ProgressCallback] = None,
) -> Any:
    """`progress_callback` (opcional) recibe los eventos {"stage", "message",
    "detail"} que emiten _node_answer/_node_enrich/deep_answer.py durante la
    corrida — el router de Telegram lo usa para ir editando un mensaje de
    estado ("Buscando…", "Pensando…") mientras el grafo trabaja. Se invoca
    SIEMPRE desde el hilo donde corre app.invoke (ver _emit_progress); si el
    caller viene de otro hilo/loop (asyncio), debe encargarse de saltar de
    vuelta de forma thread-safe."""
    # Primera señal INMEDIATA (antes de build_graph/history/clasificación): el
    # caller no debe esperar a que el grafo llegue a ninguna etapa para saber
    # que la consulta ya está en curso.
    _emit_progress(progress_callback, "graph_start", "📩 Analizando tu consulta…")

    app = build_graph()

    langfuse_handler = _make_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    history = get_history(session, limit=getattr(settings, "HISTORY_LIMIT", 10))
    add_message(session, user_id, "user", question)

    # `via` (text/image/voice/suggestion) viene del router; lo propagamos como
    # `source` para que route_entry/answer_node sepan si el mensaje es multimodal.
    source = (metadata or {}).get("via")

    state_in = make_state(
        question,
        user_id=user_id,
        session=session,
        agent_key=agent_key,
        history=history,
        source=source,
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
                "configurable": {"progress_callback": progress_callback},
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
