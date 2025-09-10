from typing import List, Dict, Optional
from collections import OrderedDict
from app.config import settings
from app.retriever.retrieve import search
from app.agent.prompt import make_prompt
from app.agent.registry import AGENTS, AgentConfig

from openai import OpenAI
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


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    agent_key: Optional[str] = None,
    chat_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Preferencias (prioridad):
      1) agent_key (usa config del agente: colección, modelo, prompt, top_k...)
      2) parámetros explícitos (collection_name, chat_model, system_prompt, top_k)
      3) settings por defecto (.env)
    """
    agent: Optional[AgentConfig] = AGENTS.get(agent_key) if agent_key else None

    coll = collection_name or (agent.collection if agent else settings.COLLECTION_NAME)
    tk   = top_k or (agent.resolved_top_k() if agent else settings.TOP_K)
    model= chat_model or (agent.resolved_chat_model() if agent else settings.CHAT_MODEL)
    sys  = system_prompt or (agent.system_prompt if agent and agent.system_prompt else "Eres un asistente útil.")

    raw_hits: List[Dict] = search(question, top_k=tk, collection_name=coll)
    hits = _dedupe_hits(raw_hits)

    prompt = make_prompt(hits, question)
    resp = _client.chat.completions.create(
        model=model,
        temperature=(agent.temperature if agent else 0.0),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        max_tokens=(agent.resolved_max_tokens() if agent else settings.MAX_TOKENS),
    )
    answer = resp.choices[0].message.content.strip()

    fuentes = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {})
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        fuentes.append(f"[{i}] {src}, p{page}")

    # Debug opcional: muestra qué agente/modelo/colección respondió
    debug = f"(Agente: {agent_key or '-'} | Modelo: {model} | Colección: {coll})"
    return f"{debug}\n\n{answer}\n\nFuentes:\n" + "\n".join(fuentes)

# Ajustar identificador para detectar el tipo de pregunta, dependiendo de esto se asignará un agente y este 
# hará el mismo flujo para devolverle la respuesta al orchestrator