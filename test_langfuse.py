# test_langfuse.py
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Inicializar cliente Langfuse (usa las env vars que ya configuraste)
lf = Langfuse()
print("Auth check:", lf.auth_check())  # debería ser True

# Configurar un LLM de ejemplo (puedes usar tu modelo real)
llm = ChatOpenAI(model="gpt-4o-mini")

# Definir estado del grafo
from typing import TypedDict
class S(TypedDict):
    user_msg: str
    reply: str

# Nodo que llama al LLM
def call_llm(state: S, config) -> S:
    resp = llm.invoke([{"role": "user", "content": state["user_msg"]}], config=config)
    return {"reply": resp.content, **state}

# Construir grafo mínimo
g = StateGraph(S)
g.add_node("llm_node", call_llm)
g.add_edge(START, "llm_node")
g.add_edge("llm_node", END)
graph = g.compile()

if __name__ == "__main__":
    lf_handler = CallbackHandler()  # callback de Langfuse
    cfg = {
        "callbacks": [lf_handler],
        "tags": ["test", "langfuse"],
        "metadata": {"component": "test_graph"},
        "configurable": {"session_id": "user-123"},
    }
    out = graph.invoke({"user_msg": "Hola, respóndeme en una sola frase"}, cfg)
    print("Respuesta del LLM:", out["reply"])
