# app/agent/graph.py
from typing import Dict, List, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler
from langchain_core.runnables.config import RunnableConfig

from app.config import settings
from app.retriever.retrieve import search
from app.agent.prompt import make_prompt
from app.llm.provider import chat_answer


# ---- State definition ----
class GraphState(TypedDict):
    question: str
    hits: List[Dict]
    prompt: str
    answer: str
    route: Literal["extractive_bot", "openai_bot"]


# ---- Nodes ----
def node_retrieve(state: GraphState) -> GraphState:
    q = state["question"]
    hits = search(q, top_k=settings.TOP_K)
    prompt = make_prompt(hits, q)
    state["hits"] = hits
    state["prompt"] = prompt
    return state


def node_router(state: GraphState) -> GraphState:
    """Decide qué 'bot' usará la respuesta.
    - extractive_bot: sin LLM (CHAT_PROVIDER=extractive)
    - openai_bot: usa chat LLM (CHAT_PROVIDER=openai)
    """
    provider = getattr(settings, "CHAT_PROVIDER", "extractive").lower()
    state["route"] = "openai_bot" if provider == "openai" else "extractive_bot"
    return state


def node_answer(state: GraphState) -> GraphState:
    ans = chat_answer(state["prompt"], state["hits"], state["question"])
    state["answer"] = ans
    return state


# ---- Graph build ----
def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("router", node_router)
    g.add_node("answer_node", node_answer)   # 👈 renombrado

    # Flujo: retrieve -> router -> answer_node -> END
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "router")
    g.add_edge("router", "answer_node")
    g.add_edge("answer_node", END)

    return g.compile()



# ---- Public API ----
_graph = None


def run_graph(question: str, user_id: str = "anon", session: str = "default") -> str:
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial: GraphState = {
        "question": question,
        "hits": [],
        "prompt": "",
        "answer": "",
        "route": "extractive_bot"
    }

    # 👇 Callback Langfuse
    lf_handler = CallbackHandler()
    cfg: RunnableConfig = {
        "callbacks": [lf_handler],
        "tags": ["retie-agent", "graph"],
        "metadata": {
            "component": "agent_graph",
            "user_id": user_id,        # id del usuario (Telegram user id)
            "session": session,        # id del chat (ej: telegram-chat-1234)
            "user_question": question  # guarda la pregunta original
        },
        "configurable": {
            "user_id": user_id,        # se pasa a Langfuse como parte del trace
            "session_id": session      # importante para agrupar por chat
        },
    }

    final_state = _graph.invoke(initial, cfg)
    return final_state["answer"]
