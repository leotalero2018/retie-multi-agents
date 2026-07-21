# app/agent/retie_agent.py
# Single, minimal agent with LLM + Chroma retrieve.
# Backward-compatible wrappers: answer_question(), run_graph()

from __future__ import annotations
from typing import List, Dict, Optional, Any
from collections import OrderedDict

from openai import OpenAI

from retie_agent.config import settings
from retie_agent.llm.provider import create_chat_completion
from retie_agent.retriever.retrieve import search
from retie_agent.agent.prompt import make_prompt

try:
    # Optional registry; if missing, we still work with defaults
    from retie_agent.agent.registry import AGENTS
except Exception:
    AGENTS = {}

# Single shared OpenAI client
_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


# --------------------------- internals --------------------------- #
def _hit_score(h: Dict) -> float:
    """Score numérico comparable. Los hits de expansión de página
    (retrieval_method="page_expand") traen score=None — se tratan como el peor
    score posible para que nunca desplacen a un hit con distancia real y para
    que None < None no reviente el dedupe (bug destapado por el deep agent al
    unir hits iniciales + acumulados por tools)."""
    s = h.get("score")
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    return float("inf")


def _dedupe_hits(hits: List[Dict]) -> List[Dict]:
    """
    Deduplicate by (source, page), keep best score.
    Compatible with your previous implementation.
    """
    best: dict[tuple, Dict] = {}
    for h in hits:
        meta = h.get("meta", {})
        key = (meta.get("source", "?"), meta.get("page", None))
        if key not in best or _hit_score(h) < _hit_score(best[key]):
            best[key] = h

    ordered = OrderedDict()
    for h in hits:
        key = (h.get("meta", {}).get("source", "?"), h.get("meta", {}).get("page", None))
        if key in best and key not in ordered:
            ordered[key] = best[key]

    return list(ordered.values())


def _resolve_collection(agent_key: Optional[str], explicit: Optional[str]) -> str:
    """
    Prefer explicit; else registry; else simple mapping; else settings default.
    """
    if explicit:
        return explicit

    if agent_key and agent_key in AGENTS and getattr(AGENTS[agent_key], "collection", None):
        return AGENTS[agent_key].collection

    # Sin agente explícito: consultar TODAS las colecciones por defecto
    # (norma_vigente + normas_historicas). search() admite nombres separados
    # por coma y cae a las colecciones existentes si ninguna coincide.
    names = getattr(settings, "COLLECTION_NAMES", "") or ""
    if names.strip():
        return names

    return getattr(settings, "COLLECTION_NAME", "retie_docs")


def _resolve_model(agent_key: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    if agent_key and agent_key in AGENTS:
        try:
            return AGENTS[agent_key].resolved_chat_model()
        except Exception:
            pass

    return getattr(settings, "CHAT_MODEL", "gpt-4o-mini")


# --------------------------- public class --------------------------- #
class RetieAgent:
    """
    Minimal orchestrator:
      - retrieve (Chroma via app.retriever.retrieve.search)
      - dedupe
      - prompt + LLM answer
    """

    def __init__(self):
        # keep constructor light; settings drive everything
        pass

    def answer(
        self,
        question: str,
        *,
        agent_key: Optional[str] = None,
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        chat_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_admin: bool = False,
    ) -> str:
        coll = _resolve_collection(agent_key, collection_name)
        model = _resolve_model(agent_key, chat_model)
        tk = top_k or getattr(settings, "TOP_K", 5)
        sys = system_prompt or "Eres un asistente útil."

        # Retrieve + dedupe
        raw_hits = search(question, top_k=tk, collection_name=coll)
        hits = _dedupe_hits(raw_hits)

        header = f"(Agente: {agent_key or '-'} | Modelo: {model} | Colección: {coll})"

        if not hits:
            return (header + "\n\n" if is_admin else "") + "No tengo evidencia en los documentos."

        # Build prompt (admin vs user)
        prompt = make_prompt(hits, question, is_admin=is_admin)

        # Call LLM
        resp = create_chat_completion(
            _client,
            model=model,
            temperature=0.0,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            max_tokens=getattr(settings, "MAX_TOKENS", 512),
        )
        answer = (resp.choices[0].message.content or "").strip()

        if not is_admin:
            return answer

        # Admin view: include explicit sources
        fuentes = []
        for i, h in enumerate(hits, 1):
            meta = h.get("meta", {})
            src = meta.get("source", "?")
            page = meta.get("page", "?")
            fuentes.append(f"[{i}] {src}, p{page}")
        return f"{header}\n\n{answer}\n\nFuentes:\n" + "\n".join(fuentes)


# ---------------------- backward-compatible API ---------------------- #
# Your existing code might import these symbols, so keep them here.

# Single module-level agent (stateless enough to reuse)
_AGENT_SINGLETON: Optional[RetieAgent] = None


def _agent() -> RetieAgent:
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = RetieAgent()
    return _AGENT_SINGLETON


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    agent_key: Optional[str] = None,
    chat_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    *,
    is_admin: bool = False,
) -> str:
    """
    Drop-in replacement for app.agent.orchestrator.answer_question
    """
    return _agent().answer(
        question,
        agent_key=agent_key,
        top_k=top_k,
        collection_name=collection_name,
        chat_model=chat_model,
        system_prompt=system_prompt,
        is_admin=is_admin,
    )


def run_graph(
    question: str,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
) -> str:
    """
    Drop-in replacement for app.agent.graph.run_graph
    (No LangGraph; we simply call the agent directly.)
    """
    # We still pass through agent_key so your per-collection routing remains intact.
    return answer_question(question, agent_key=agent_key, is_admin=False)
