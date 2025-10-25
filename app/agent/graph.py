# app/agent/graph.py
from __future__ import annotations

from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from openai import OpenAI

from app.config import settings
from app.retriever.retrieve import search
from app.agent.prompt import make_prompt
from app.agent.retie_agent import (
    _dedupe_hits,
    _resolve_collection,
    _resolve_model,
)
from app.observability.obs import trace_ctx, span_ctx, log_generation


# ---------- State ----------

class GraphState(TypedDict, total=False):
    question: str
    user_id: str
    session: str
    agent_key: Optional[str]
    hits: List[Dict[str, Any]]
    route: str
    answer: str


def make_state(
    question: str,
    *,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
) -> GraphState:
    return {
        "question": (question or "").strip(),
        "user_id": user_id or "anon",
        "session": session or "default",
        "agent_key": agent_key,
    }


# ---------- Low-level LLM client (re-usa tu config) ----------

_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


# ---------- Nodes ----------

def _node_retrieve(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")
    coll = _resolve_collection(agent_key, explicit=None)
    top_k = getattr(settings, "TOP_K", 4)

    # Cada nodo abre su span para Langfuse
    with span_ctx(None, "retrieve", {"q": q, "collection": coll, "top_k": top_k}):
        hits = _dedupe_hits(search(q, top_k=top_k, collection_name=coll))
        return {"hits": hits}


def _node_router(state: GraphState) -> GraphState:
    hits = state.get("hits") or []
    route = "answer_node" if len(hits) > 0 else "no_context"
    with span_ctx(None, "router", {"hits": len(hits), "route": route}):
        return {"route": route}


def _node_answer(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")

    hits = state.get("hits") or []
    if not hits:
        # Normalmente no llegamos aquí si router decide "no_context", pero lo dejamos defensivo
        return {"answer": "No tengo evidencia en los documentos."}

    model = _resolve_model(agent_key, explicit=None)
    sys = "Eres un asistente útil."
    prompt = make_prompt(hits, q, is_admin=False)

    with span_ctx(None, "answer_node", {"model": model, "hits": len(hits)}):
        resp = _client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=getattr(settings, "MAX_TOKENS", 600),
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()

        # Guarda la generación en la traza (aparece como "Generation" en Langfuse)
        try:
            usage = getattr(resp, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            log_generation(
                None,
                name="openai.chat",
                input_text=prompt,
                output_text=answer,
                model=model,
                usage=usage_dict,
                metadata={"agent_key": agent_key},
            )
        except Exception:
            pass

        return {"answer": answer}


def _node_no_context(state: GraphState) -> GraphState:
    with span_ctx(None, "answer_node", {"route": "no_context"}):
        return {"answer": "No tengo evidencia en los documentos."}


# ---------- Graph builder (singleton compilado) ----------

@dataclass
class _Compiled:
    app: Any


_COMPILED: Optional[_Compiled] = None


def build_graph():
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED.app

    g = StateGraph(GraphState)

    g.add_node("retrieve", _node_retrieve)
    g.add_node("router", _node_router)
    g.add_node("answer_node", _node_answer)
    g.add_node("no_context", _node_no_context)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "router")
    g.add_conditional_edges(
        "router",
        lambda s: s.get("route", "no_context"),
        {
            "answer_node": "answer_node",
            "no_context": "no_context",
        },
    )
    g.add_edge("answer_node", END)
    g.add_edge("no_context", END)

    app = g.compile()
    _COMPILED = _Compiled(app=app)
    return app


# ---------- Public runner ----------

def run_graph(
    question: str,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Ejecuta el grafo con instrumentación Langfuse. Devuelve sólo el texto final.
    """
    # IMPORTANTE: nombramos la traza raíz "LangGraph" para que en Langfuse
    # el listado salga exactamente como en la captura.
    with trace_ctx(
        name="LangGraph",
        user_id=user_id,
        metadata={
            "component": "agent_graph",
            "tags": ["retie-agent", "graph"],
            "session_id": session,
            **(metadata or {}),
        },
    ):
        app = build_graph()
        out = app.invoke(make_state(question, user_id=user_id, session=session, agent_key=agent_key))
        return out.get("answer", "No tengo evidencia en los documentos.")
