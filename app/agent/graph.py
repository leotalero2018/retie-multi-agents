# app/agent/graph.py
from typing import Dict, List, Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler
from langchain_core.runnables.config import RunnableConfig

from app.config import settings
from app.retriever.retrieve import search
from app.agent.prompt import make_prompt
from app.llm.provider import chat_answer
from app.agent.registry import AGENTS
from app.agent.orchestrator import _dedupe_hits


# ---- State definition ----
class GraphState(TypedDict):
    question: str
    hits: List[Dict]
    prompt: str
    answer: str
    route: Literal["extractive_bot", "openai_bot"]
    agent_key: Optional[str]


# ---- Nodes ----
def node_retrieve(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")
    collection_name: Optional[str] = None

    # Si hay agente, usamos su colección; si no, la default de settings
    if agent_key and agent_key in AGENTS:
        collection_name = AGENTS[agent_key].collection

    raw_hits = search(q, top_k=settings.TOP_K, collection_name=collection_name)
    hits = _dedupe_hits(raw_hits)
    state["hits"] = hits
    state["prompt"] = make_prompt(hits, q)
    return state


def node_router(state: GraphState) -> GraphState:
    """Decide qué 'bot' usar"""
    provider = getattr(settings, "CHAT_PROVIDER", "extractive").lower()
    state["route"] = "openai_bot" if provider == "openai" else "extractive_bot"
    return state


def node_build_prompt(state: GraphState) -> GraphState:
    q = state["question"]
    hits = state["hits"]
    state["prompt"] = make_prompt(hits, q)
    return state


def node_answer(state: GraphState) -> GraphState:
    ans = chat_answer(state["prompt"], state["hits"], state["question"])
    state["answer"] = ans
    return state


# ---- Graph build ----
def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # Añadimos nodos
    g.add_node("retrieve", node_retrieve)
    g.add_node("router", node_router)
    g.add_node("build_prompt", node_build_prompt)
    g.add_node("answer", node_answer)

    # Definimos flujo
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "router")
    g.add_edge("router", "build_prompt")
    g.add_edge("build_prompt", "answer")
    g.add_edge("answer", END)

    return g.compile()


# ---- Public API ----
_graph: Optional[StateGraph] = None


def run_graph(
    question: str,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
) -> str:
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: GraphState = {
        "question": question,
        "hits": [],
        "prompt": "",
        "answer": "",
        "route": "extractive_bot",
        "agent_key": agent_key,
    }

    # Configuración de Langfuse Callback
    lf_handler = CallbackHandler()
    cfg: RunnableConfig = {
        "callbacks": [lf_handler],
        "tags": ["retie-agent", "graph"] + ([f"agent:{agent_key}"] if agent_key else []),
        "metadata": {
            "component": "agent_graph",
            "user_id": user_id,
            "session": session,
            "user_question": question,
            "agent_key": agent_key,
        },
        "configurable": {
            "user_id": user_id,
            "session_id": session,
        },
    }

    final_state = _graph.invoke(initial_state, cfg)
    return final_state["answer"]
