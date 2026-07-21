# retie_agent/agent/deep_answer.py
"""Deep agent del answer_node (Fase 4 del plan v3 — plans/02 + FINAL_IMPLEMENTATION_NODES §5).

Loop ReAct acotado (deepagents.create_deep_agent) que:
  1. evalúa la evidencia inicial del fan-in (Chroma + fuente secundaria, que
     sigue llegando gratis de las ramas paralelas del grafo);
  2. si basta, responde directo (una pasada, costo similar al camino clásico);
  3. si falta, re-consulta las fuentes con queries refinadas — tools
     `search_chroma` y `ask_gemini` (NotebookLM NO es tool: 30–180 s por llamada
     vía navegador romperían el presupuesto; su aporte entra como evidencia
     inicial) — iterando con límites duros (timeout / recursion / nº de
     re-consultas);
  4. sintetiza fiel a la evidencia según la skill que el registry determinista
     (skills.py) resolvió a partir del IntentV3 — el agente NO elige su skill.

Contención de middleware (verificada en el spike Fase 0): el HarnessProfile
"openai" excluye los built-ins de deepagents (write_todos, filesystem, execute,
task) y el TodoListMiddleware; Filesystem/SubAgentMiddleware son scaffolding no
excluible pero sus tools quedan fuera de la vista del modelo.

NO definir LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 en ningún entorno:
deepagents arrastra langsmith y la observabilidad de este proyecto es Langfuse.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from retie_agent.config import settings
from retie_agent.agent.prompt import SYSTEM_PROMPT_RETIE, build_context
from retie_agent.agent.skills import SkillSpec, resolve_skill

# Pool propio: NO importar _POOL de graph.py (evitaría un import circular).
_DEEP_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="deep")

# Frase canónica compartida con el grafo: el agente la emite cuando no hay
# evidencia suficiente y _node_answer la convierte en route="no_context".
NO_EVIDENCE_PHRASE = "No tengo evidencia en los documentos."


# ──────────────────────────────────────────────────────────────────────────────
# Contención de middleware de deepagents (decisión del spike Fase 0.4)
# ──────────────────────────────────────────────────────────────────────────────

_PROFILE_REGISTERED = False


def _ensure_harness_profile() -> None:
    """Registra (una vez por proceso) el perfil que apaga los built-ins.

    Sin esto, deepagents añade write_todos / ls / read_file / write_file /
    edit_file / glob / grep / execute / task al loop — tools irrelevantes que
    solo queman tokens y confunden al modelo.
    """
    global _PROFILE_REGISTERED
    if _PROFILE_REGISTERED:
        return
    from deepagents import HarnessProfile, register_harness_profile

    register_harness_profile(
        "openai",
        HarnessProfile(
            excluded_tools=frozenset({
                "write_todos", "ls", "read_file", "write_file", "edit_file",
                "glob", "grep", "execute", "task",
            }),
            excluded_middleware=frozenset({"TodoListMiddleware"}),
        ),
    )
    _PROFILE_REGISTERED = True


# ──────────────────────────────────────────────────────────────────────────────
# Evidencia acumulada por las tools (per-request → thread-safe por construcción)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceLog:
    chroma_hits: List[Dict[str, Any]] = field(default_factory=list)
    gemini_sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # {tool, query, n_results[, error]}


ProgressCallback = Callable[[Dict[str, Any]], None]


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    message: str,
    **detail: Any,
) -> None:
    """Emite progreso best-effort para superficies interactivas.

    El deep agent corre dentro de un thread; por eso el callback debe ser
    sincrónico y thread-safe desde el caller (Telegram usa loop.call_soon_threadsafe).
    Nunca debe afectar la respuesta si falla.
    """
    if progress_callback is None:
        return
    try:
        progress_callback({"stage": stage, "message": message, "detail": detail})
    except Exception:
        pass


# Cliente Gemini propio y perezoso (no reusar el de graph.py → import circular).
_gemini_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from retie_agent.agent.gemini_client import GeminiFileSearchClient

        _gemini_client = GeminiFileSearchClient(
            api_key=getattr(settings, "GEMINI_API_KEY", None),
            model=getattr(settings, "GEMINI_MODEL", "gemini-flash-latest"),
            store=getattr(settings, "GEMINI_FILE_SEARCH_STORE", None),
            timeout=float(getattr(settings, "GEMINI_TIMEOUT", 120.0)),
        )
    return _gemini_client


def _build_tools(
    log: EvidenceLog,
    agent_key: Optional[str],
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Any]:
    """Construcción DINÁMICA de tools según qué fuentes estén disponibles.

    Contrato de tool en este proyecto: firma tipada simple, salida SIEMPRE str
    (contexto formateado o mensaje de degradación), NUNCA lanza, y registra su
    actividad en el EvidenceLog del request (las tools cierran sobre él).
    """
    from langchain_core.tools import tool

    tools: List[Any] = []

    @tool
    def search_chroma(query: str, top_k: int = 6) -> str:
        """Busca fragmentos literales de la normativa (RETIE / NTC 2050) en el índice
        vectorial. Rinde más con referencias literales ("tabla 220.55",
        "artículo 250.24") o términos exactos del documento que con preguntas largas."""
        _emit_progress(
            progress_callback,
            "search_chroma",
            "📚 Consultando fragmentos normativos…",
            query=query,
            top_k=top_k,
        )
        try:
            from retie_agent.retriever.retrieve import search
            from retie_agent.agent.retie_agent import _resolve_collection

            hits = search(
                query, top_k=top_k,
                collection_name=_resolve_collection(agent_key, None),
            ) or []
            log.chroma_hits.extend(hits)
            log.tool_calls.append(
                {"tool": "search_chroma", "query": query, "n_results": len(hits)}
            )
            _emit_progress(
                progress_callback,
                "search_chroma_done",
                "📚 Evidencia normativa recuperada; revisando relevancia…",
                query=query,
                n_results=len(hits),
            )
            if not hits:
                return (
                    "Sin resultados para esa consulta. Prueba una formulación distinta "
                    "o una referencia literal (tabla/artículo)."
                )
            return build_context(hits, include_meta=True)
        except Exception as exc:  # noqa: BLE001 — la tool degrada, nunca lanza
            logger.warning("search_chroma tool failed: %s", exc)
            log.tool_calls.append({
                "tool": "search_chroma", "query": query,
                "n_results": 0, "error": str(exc)[:200],
            })
            _emit_progress(
                progress_callback,
                "search_chroma_error",
                "⚠️ La búsqueda normativa falló; continúo con la evidencia disponible…",
                query=query,
                error=str(exc)[:200],
            )
            return "La búsqueda falló; intenta otra query o responde con la evidencia que ya tienes."

    tools.append(search_chroma)

    client = None
    try:
        client = _get_gemini()
    except Exception:  # pragma: no cover — sin google-genai instalado
        client = None
    if client is not None and client.is_available():

        @tool
        def ask_gemini(query: str) -> str:
            """Consulta el corpus normativo completo vía Gemini File Search y devuelve una
            respuesta sintetizada. Útil cuando la búsqueda literal no encuentra la información."""
            _emit_progress(
                progress_callback,
                "ask_gemini",
                "🔎 Consultando el corpus completo para contrastar la respuesta…",
                query=query,
            )
            try:
                # Acotado al presupuesto del agente (75 s): nunca esperar los 120 s del setting.
                timeout = min(float(getattr(settings, "GEMINI_TIMEOUT", 120.0)), 45.0)
                fut = _DEEP_POOL.submit(client.ask_question, query)
                raw_answer, raw_sources = fut.result(timeout=timeout)
                answer = (raw_answer or "").strip()
                log.gemini_sources.extend(raw_sources or [])
                log.tool_calls.append(
                    {"tool": "ask_gemini", "query": query, "n_results": len(raw_sources or [])}
                )
                _emit_progress(
                    progress_callback,
                    "ask_gemini_done",
                    "🔎 Consulta complementaria recibida; cruzando fuentes…",
                    query=query,
                    n_results=len(raw_sources or []),
                )
                return answer or "Gemini no encontró información para esa consulta."
            except Exception as exc:  # noqa: BLE001
                logger.warning("ask_gemini tool failed: %s", exc)
                log.tool_calls.append({
                    "tool": "ask_gemini", "query": query,
                    "n_results": 0, "error": str(exc)[:200],
                })
                _emit_progress(
                    progress_callback,
                    "ask_gemini_error",
                    "⚠️ La consulta complementaria falló; continúo con la evidencia disponible…",
                    query=query,
                    error=str(exc)[:200],
                )
                return "La consulta a Gemini falló; continúa con la evidencia disponible."

        tools.append(ask_gemini)

    return tools


def deep_tools_available() -> bool:
    """¿Hay al menos una tool de re-consulta utilizable? (Chroma casi siempre.)

    _node_answer la usa para el corte temprano: sin evidencia inicial Y sin
    ninguna tool → no_context directo (como el grafo clásico); con tools, el
    agente corre y puede reformular/encontrar.
    """
    try:
        from retie_agent.retriever import retrieve  # noqa: F401
        return True
    except Exception:
        pass
    try:
        client = _get_gemini()
        return bool(client is not None and client.is_available())
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Modelo y prompts
# ──────────────────────────────────────────────────────────────────────────────

def _chat_model_for(model_name: str, max_tokens: int):
    """ChatOpenAI con kwargs seguros por familia.

    gpt-5*/o-series rechazan `temperature`≠1 y `max_tokens` (exigen
    `max_completion_tokens`), y ChatOpenAI NO tiene la auto-recuperación de
    create_chat_completion — hay que acertar a la primera. Se usa
    chat.completions (use_responses_api=False) como el resto del código.
    """
    from langchain_openai import ChatOpenAI

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "api_key": getattr(settings, "OPENAI_API_KEY", None) or "sk-missing",
        "use_responses_api": False,
    }
    low = (model_name or "").lower()
    reasoning_family = low.startswith(("gpt-5", "o1", "o3", "o4"))
    if reasoning_family:
        # gpt-5*/o-series: sin temperature (solo aceptan la default) y presupuesto
        # vía max_completion_tokens (param nativo de ChatOpenAI; max_tokens da 400).
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


DEEP_AGENT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT_RETIE
    + "\n\nPROCEDIMIENTO:\n"
    "1. Evalúa la EVIDENCIA INICIAL del mensaje: si basta para responder con "
    "precisión, responde DIRECTO sin usar herramientas (no busques por buscar).\n"
    "2. Si falta información, re-consulta con queries REFINADAS y DISTINTAS a la "
    "original: referencias literales (\"tabla 220.55\", \"artículo 250.24\"), "
    "sinónimos normativos, términos exactos del documento. Máximo {max_retrievals} "
    "re-consultas en total.\n"
    "3. Sé fiel a la evidencia: no inventes valores, artículos ni numerales; no "
    "redondees ni omitas.\n"
    "4. Si tras buscar no hay evidencia suficiente, responde exactamente: "
    "\"" + NO_EVIDENCE_PHRASE + "\"\n"
    "5. No menciones herramientas, búsquedas ni tu proceso en la respuesta final."
)


def _build_initial_message(
    question: str,
    initial_hits: List[Dict[str, Any]],
    secondary_answer: str,
    skill: SkillSpec,
    intent_meta: Optional[Dict[str, Any]],
    wants_full: bool,
) -> str:
    """Mensaje inicial: evidencia del fan-in (el agente NO repite el retrieval
    inicial) + directivas resueltas por el registry + parámetros del classifier."""
    parts: List[str] = []

    if initial_hits:
        parts.append(
            "FRAGMENTOS DEL DOCUMENTO (evidencia inicial ya recuperada):\n"
            + build_context(initial_hits, include_meta=True)
        )
    else:
        parts.append("FRAGMENTOS DEL DOCUMENTO: (ninguno recuperado en la pasada inicial)")

    if secondary_answer:
        parts.append("ANÁLISIS COMPLEMENTARIO (fuente secundaria):\n" + secondary_answer)

    directives: List[str] = [skill.directives]
    meta = intent_meta or {}
    entities = meta.get("entities") or []
    if entities:
        refs = ", ".join(e.get("raw") or e.get("value", "") for e in entities if isinstance(e, dict))
        if refs:
            directives.append(
                f"Referencias detectadas (candidatas a búsqueda literal): {refs}."
            )
    if meta.get("complexity") == "high":
        directives.append(
            "El mensaje contiene varias preguntas: descomponlo en sub-preguntas y haz "
            "una re-consulta dirigida por cada una (respetando el tope)."
        )
    if meta.get("needs_calculation"):
        directives.append(
            "Muestra el cálculo paso a paso usando SOLO valores y factores presentes "
            "en la evidencia; si falta un factor, dilo en vez de inventarlo."
        )
    conf = meta.get("confidence")
    if isinstance(conf, (int, float)) and conf < 0.5:
        directives.append(
            "La clasificación de la consulta es dudosa: verifica con una re-consulta "
            "antes de afirmar."
        )
    if wants_full and skill.name != "exhaustive":
        directives.append(DIR_FULL_NOTE)

    parts.append("DIRECTIVAS DE ENTREGA:\n- " + "\n- ".join(directives))
    parts.append("PREGUNTA DEL USUARIO:\n" + question)
    return "\n\n".join(parts)


DIR_FULL_NOTE = (
    "Respuesta COMPLETA: enumera todos los literales/numerales presentes en la "
    "evidencia y declara explícitamente los faltantes."
)


def _content_to_text(content: Any) -> str:
    """Normaliza el content de un AIMessage (str o lista de bloques) a texto."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return str(content or "")


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DeepAnswerResult:
    answer: str
    hits: List[Dict[str, Any]]      # SOLO los acumulados por las tools (el caller une)
    status: str                     # "ok" | "fallback_timeout" | "fallback_error"
    tool_calls: List[Dict[str, Any]]
    usage: Dict[str, Optional[int]]
    skill: str = "qa"


def run_deep_answer(
    question: str,
    *,
    initial_hits: List[Dict[str, Any]],
    secondary_answer: str,
    history: Optional[List[Dict[str, str]]] = None,
    agent_key: Optional[str] = None,
    intent: Optional[str] = None,
    intent_meta: Optional[Dict[str, Any]] = None,
    wants_table: bool = False,
    wants_full: bool = False,
    model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> DeepAnswerResult:
    """Corre el deep agent para una consulta. Nunca lanza: timeout/error →
    status de fallback con answer="" (el caller degrada a _synthesize_simple).

    `config` debe traer el RunnableConfig del nodo (con el CallbackHandler de
    Langfuse): el agente corre en otro thread y sin config explícito la traza
    del subgrafo se corta.
    """
    log = EvidenceLog()
    skill = resolve_skill(intent, (intent_meta or {}).get("response_format"))

    base_tokens = int(getattr(settings, "DEEP_AGENT_MAX_TOKENS", 1600))
    max_tokens = 4000 if (wants_table or wants_full) else max(base_tokens, skill.max_tokens)
    model_name = (
        getattr(settings, "DEEP_AGENT_MODEL", None)
        or model
        or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    )
    timeout = float(getattr(settings, "DEEP_AGENT_TIMEOUT", 75.0))
    recursion_limit = int(getattr(settings, "DEEP_AGENT_RECURSION_LIMIT", 12))
    max_retrievals = int(getattr(settings, "DEEP_AGENT_MAX_RETRIEVALS", 4))

    answer = ""
    status = "ok"
    usage: Dict[str, Optional[int]] = {"input": None, "output": None, "total": None}
    try:
        _emit_progress(
            progress_callback,
            "deep_answer_start",
            "🧠 Revisando la evidencia inicial y decidiendo si hace falta buscar más…",
            intent=intent,
            skill=skill.name,
            initial_hits=len(initial_hits),
            has_secondary=bool((secondary_answer or "").strip()),
        )
        # TODO dentro del try — incluidos los imports de deepagents/langchain
        # (_ensure_harness_profile, _build_tools): un entorno con el cluster
        # desalineado debe degradar a _synthesize_simple, no romper la respuesta.
        _ensure_harness_profile()
        tools = _build_tools(log, agent_key, progress_callback=progress_callback)
        system_prompt = DEEP_AGENT_SYSTEM_PROMPT.format(max_retrievals=max_retrievals)
        initial_message = _build_initial_message(
            question, initial_hits, secondary_answer, skill, intent_meta, wants_full
        )

        def _invoke():
            import deepagents

            agent = deepagents.create_deep_agent(
                model=_chat_model_for(model_name, max_tokens),
                tools=tools,
                system_prompt=system_prompt,
            )
            messages: List[Dict[str, str]] = [
                {"role": m["role"], "content": m["content"]}
                for m in (history or [])
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            messages.append({"role": "user", "content": initial_message})
            cfg = dict(config or {})
            cfg["recursion_limit"] = recursion_limit
            cfg["run_name"] = "deep_answer_agent"
            _emit_progress(
                progress_callback,
                "deep_agent_reasoning",
                "⚙️ Analizando requisitos, fuentes y posibles vacíos…",
                max_retrievals=max_retrievals,
            )
            return agent.invoke({"messages": messages}, cfg)

        result = _DEEP_POOL.submit(_invoke).result(timeout=timeout)
        _emit_progress(
            progress_callback,
            "deep_agent_synthesizing",
            "✍️ Sintetizando la respuesta con la evidencia encontrada…",
            tool_calls=len(log.tool_calls),
        )
        msgs = (result or {}).get("messages") or []
        for m in reversed(msgs):
            if getattr(m, "type", None) == "ai":
                text = _content_to_text(getattr(m, "content", "")).strip()
                if text:
                    answer = text
                    break
        tin = tout = 0
        seen_usage = False
        for m in msgs:
            um = getattr(m, "usage_metadata", None)
            if isinstance(um, dict):
                seen_usage = True
                tin += um.get("input_tokens") or 0
                tout += um.get("output_tokens") or 0
        if seen_usage:
            usage = {"input": tin, "output": tout, "total": tin + tout}
        if not answer:
            status = "fallback_error"
            logger.warning("deep agent devolvió respuesta vacía — se degrada a síntesis simple")
        else:
            _emit_progress(
                progress_callback,
                "deep_answer_done",
                "✅ Respuesta técnica lista; preparando presentación…",
                answer_len=len(answer),
                tool_calls=len(log.tool_calls),
            )
    except FutureTimeoutError:
        status = "fallback_timeout"
        _emit_progress(
            progress_callback,
            "deep_answer_timeout",
            "⏱️ La revisión profunda tardó demasiado; preparo una respuesta de respaldo…",
            timeout=timeout,
        )
        logger.warning(
            "deep agent agotó el presupuesto de %.0fs — se degrada a síntesis simple", timeout
        )
    except Exception as exc:  # noqa: BLE001 — el nodo decide el fallback
        status = "fallback_error"
        _emit_progress(
            progress_callback,
            "deep_answer_error",
            "⚠️ La revisión profunda falló; preparo una respuesta de respaldo…",
            error=str(exc)[:200],
        )
        logger.warning("deep agent falló: %s", exc)

    return DeepAnswerResult(
        answer=answer,
        hits=list(log.chroma_hits),
        status=status,
        tool_calls=list(log.tool_calls),
        usage=usage,
        skill=skill.name,
    )
