# app/agent/graph.py
# Backward-compatible shim: remove LangGraph dependency.
from typing import Optional
from .retie_agent import run_graph as _run_graph

def run_graph(
    question: str,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
) -> str:
    return _run_graph(question, user_id=user_id, session=session, agent_key=agent_key)
