# app/agent/orchestrator.py
from typing import List, Dict, Optional
from collections import OrderedDict
from openai import OpenAI

from app.config import settings
from app.retriever.retrieve import search
from app.agent.prompt import make_prompt

try:
    from app.agent.registry import AGENTS, AgentConfig
except Exception:
    AGENTS = {}
    AgentConfig = object  # placeholder

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def _dedupe_hits(hits: List[Dict]) -> List[Dict]:
    best: dict[tuple, Dict] = {}
    for h in hits:
        meta = h.get("meta", {})
        key = (meta.get("source", "?"), meta.get("page", None))
        if key not in best or h.get("score", 1e9) < best[key].get("score", 1e9):
            best[key] = h
    ordered = OrderedDict()
    for h in hits:
        key = (h["meta"].get("source", "?"), h["meta"].get("page", None))
        if key in best and key not in ordered:
            ordered[key] = best[key]
    return list(ordered.values())


def _resolve_collection(agent_key: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    agent = AGENTS.get(agent_key) if agent_key else None
    if agent and getattr(agent, "collection", None):
        return agent.collection
    # mapping rápido si no usas registry
    if agent_key == "plumber":
        return "retie_plumber"
    if agent_key == "pymupdf":
        return "retie_pymupdf"
    if agent_key == "hybrid":
        return "retie_hybrid"
    return settings.COLLECTION_NAME


def _resolve_model(agent_key: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    agent = AGENTS.get(agent_key) if agent_key else None
    if agent:
        try:
            return agent.resolved_chat_model()
        except Exception:
            return getattr(agent, "chat_model", settings.CHAT_MODEL)
    return settings.CHAT_MODEL


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    agent_key: Optional[str] = None,
    chat_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    coll = _resolve_collection(agent_key, collection_name)
    model = _resolve_model(agent_key, chat_model)
    tk = top_k or settings.TOP_K
    sys = system_prompt or "Eres un asistente útil."

    raw_hits = search(question, top_k=tk, collection_name=coll)
    hits = _dedupe_hits(raw_hits)

    header = f"(Agente: {agent_key or '-'} | Modelo: {model} | Colección: {coll})"

    if not hits:
        return f"{header}\n\nNo tengo evidencia en los documentos."

    prompt = make_prompt(hits, question)
    resp = _client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
        max_tokens=settings.MAX_TOKENS,
    )
    answer = resp.choices[0].message.content.strip()

    fuentes = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {})
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        fuentes.append(f"[{i}] {src}, p{page}")

    return f"{header}\n\n{answer}\n\nFuentes:\n" + "\n".join(fuentes)
