# app/agent/graph.py
from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from openai import OpenAI

from retie_agent.config import settings
from retie_agent.retriever.retrieve import search
from retie_agent.agent.prompt import make_prompt
from retie_agent.agent.retie_agent import _dedupe_hits, _resolve_collection, _resolve_model
from retie_agent.observability.obs import span_ctx, log_generation
from retie_agent.services.history import get_history, add_message

# ---------- State ----------
class GraphState(TypedDict, total=False):
    question: str
    user_id: str
    session: str
    agent_key: Optional[str]
    hits: List[Dict[str, Any]]
    route: str
    answer: Any   # may now hold dict for styled payload
    metadata: Optional[Dict[str, Any]]  # new, for passing channel info
    history: List[Dict[str, str]]


def make_state(question: str, *, user_id: str = "anon", session: str = "default", agent_key: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> GraphState:
    return {
        "question": (question or "").strip(),
        "user_id": user_id or "anon",
        "session": session or "default",
        "agent_key": agent_key,
        "history": history or [],
    }

_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))

# ---------- Nodes ----------
def _node_retrieve(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")
    coll = _resolve_collection(agent_key, explicit=None)
    top_k = getattr(settings, "TOP_K", 4)

    # Make this a "retriever" observation so Langfuse can draw the graph.
    with span_ctx(None, "retrieve", {"collection": coll, "top_k": top_k}, as_type="retriever", span_input={"q": q}):
        try:
            hits = _dedupe_hits(search(q, top_k=top_k, collection_name=coll))
        except Exception:
            hits = []
        return {"hits": hits}

def _node_router(state: GraphState) -> GraphState:
    hits = state.get("hits") or []
    route = "answer_node" if len(hits) > 0 else "no_context"
    # Mark as a generic "chain" step
    with span_ctx(None, "router", {"hits": len(hits), "route": route}, as_type="chain"):
        return {"route": route}

def _node_answer(state: GraphState) -> GraphState:
    q = state["question"]
    agent_key = state.get("agent_key")
    hits = state.get("hits") or []
    if not hits:
        return {"answer": "No tengo evidencia en los documentos."}

    model = _resolve_model(agent_key, explicit=None)
    sys = "Eres un asistente útil."
    prompt = make_prompt(hits, q, is_admin=False)
    messages = [{"role": "system", "content": sys}]
    # Add history
    for msg in state.get("history", []):
        messages.append(msg)
    # Add project context/query
    messages.append({"role": "user", "content": prompt})

    with span_ctx(None, "answer_node", {"model": model, "hits": len(hits)}, as_type="chain"):
        resp = _client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=getattr(settings, "MAX_TOKENS", 600),
            messages=messages,
        )
        answer = (resp.choices[0].message.content or "").strip()

        try:
            usage = getattr(resp, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            log_generation(None, name="openai.chat", input_text=prompt, output_text=answer, model=model, usage=usage_dict, metadata={"agent_key": agent_key})
        except Exception:
            pass

        return {"answer": answer}

def _node_no_context(_state: GraphState) -> GraphState:
    with span_ctx(None, "answer_node", {"route": "no_context"}, as_type="chain"):
        return {"answer": "No tengo evidencia en los documentos."}


from retie_agent.agent.enrichment_assistant import EnrichmentAssistant
import os
from retie_agent.observability.obs import span_ctx, log_generation  # ✅ keep logs visible in Langfuse

# ===============================
#  Enrichment Assistant Settings
# ===============================
ENRICHMENT_ASSISTANT_ID = os.getenv(
    "ENRICHMENT_ASSISTANT_ID", "asst_qthy1ZfTpr2ps0mruX30zVlc"
)
ENRICHMENT_VECTOR_STORE_ID = os.getenv(
    "ENRICHMENT_VECTOR_STORE_ID", "vs_69050fe6e43c8191be28bac47c3f565f"
)

_enrichment_agent = EnrichmentAssistant(
    assistant_id=ENRICHMENT_ASSISTANT_ID,
    vector_store_id=ENRICHMENT_VECTOR_STORE_ID,
)

# ===============================
#  Enrichment Node
# ===============================
def _node_enrich(state: GraphState) -> GraphState:
    """Asistente secundario que enriquece la respuesta base sin bloquear el flujo."""
    base_answer = state.get("answer", "")
    question = state.get("question", "")

    with span_ctx(
        None,
        "enrich_node",
        {"assistant_id": ENRICHMENT_ASSISTANT_ID},
        as_type="chain",
    ):
        try:
            # Intento de enriquecimiento con el asistente secundario
            enriched = _enrichment_agent.enrich_response(
                user_message=question,
                draft_response=base_answer,
            )

            # Si el enriquecimiento no produce texto, usa la respuesta base
            if not enriched or len(enriched.strip()) < 20:
                raise ValueError("Sin resultados de enriquecimiento o vector vacío")

            status = "ok"

        except Exception as e:
            # Si falla (por vector vacío o error en la API), se conserva la respuesta original
            enriched = f"{base_answer}\n\n(ℹ️ Enriquecimiento omitido: {e})"
            status = "fallback"

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
                },
            )
        except Exception:
            pass

        return {"answer": enriched}


# ===============================
#  Stylist Node (Output Formatter)
# ===============================
def _node_stylist(state: GraphState) -> GraphState:
    """
    Post-processing node that prepares the enriched response for downstream delivery.

    It can adapt tone or format depending on the output channel (Telegram, WhatsApp, Web, etc.).
    The output is a JSON-ready structure, making it easy to add or swap channels later.
    """
    enriched_answer = state.get("answer", "")
    question = state.get("question", "")
    metadata = state.get("metadata", {}) or {}
    channel = metadata.get("channel", "telegram")  # default channel

    with span_ctx(
        None,
        "stylist_node",
        {"channel": channel},
        as_type="chain",
    ):
        try:
            # Simple rules: add emojis, Markdown, or remove unsupported tags depending on channel
            if channel == "telegram":
                styled_text = enriched_answer  # Telegram already supports Markdown/HTML
            elif channel == "whatsapp":
                styled_text = enriched_answer.replace("*", "").replace("_", "")
            elif channel == "web":
                styled_text = f"<p>{enriched_answer}</p>"
            else:
                styled_text = enriched_answer

            # Unified JSON payload (future-proof)
            styled_payload = {
                "channel": channel,
                "user_message": question,
                "formatted_response": styled_text,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            }

            status = "ok"

        except Exception as e:
            styled_payload = {
                "channel": channel,
                "user_message": question,
                "formatted_response": enriched_answer,
                "error": str(e),
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            }
            status = "fallback"

        # Log to Langfuse for visibility
        try:
            log_generation(
                None,
                name="stylist_node",
                input_text=question,
                output_text=str(styled_payload),
                model="stylist_formatter",
                metadata={"channel": channel, "status": status},
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
    g.add_node("retrieve", _node_retrieve)
    g.add_node("router", _node_router)
    g.add_node("answer_node", _node_answer)
    g.add_node("enrich_node", _node_enrich)
    g.add_node("stylist_node", _node_stylist)
    g.add_node("no_context", _node_no_context)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "router")

    g.add_conditional_edges(
        "router",
        lambda s: s.get("route", "no_context"),
        {"answer_node": "answer_node", "no_context": "no_context"},
    )

    # New sequence: answer → enrich → stylist → end
    g.add_edge("answer_node", "enrich_node")
    g.add_edge("enrich_node", "stylist_node")
    g.add_edge("stylist_node", END)
    g.add_edge("no_context", END)

    app = g.compile()
    _COMPILED = _Compiled(app=app)
    return app


# ---------- Runner ----------
from langfuse.langchain import CallbackHandler

def run_graph(
    question: str,
    user_id: str = "anon",
    session: str = "default",
    agent_key: Optional[str] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    app = build_graph()

    langfuse_handler = CallbackHandler()

    history = get_history(session, limit=getattr(settings, "HISTORY_LIMIT", 10))
    add_message(session, user_id, "user", question)

    state_in = make_state(
        question,
        user_id=user_id,
        session=session,
        agent_key=agent_key,
        history=history,
    )

    # invoke() devuelve el estado final directamente; evita el bug
    # "generator didn't stop after throw()" que ocurría con stream() + break
    result: Dict[str, Any] = app.invoke(
        state_in,
        config={
            "callbacks": [langfuse_handler],
            "run_name": "LangGraph",
            "tags": ["retie-agent", "graph", f"user:{user_id}", f"session:{session}"],
            "metadata": {**(metadata or {}), "agent_key": agent_key},
        },
    )

    final_answer = result.get("answer") if isinstance(result, dict) else None
    final_txt = final_answer or "No tengo evidencia en los documentos."

    txt_to_save = final_txt
    if isinstance(final_txt, dict) and "formatted_response" in final_txt:
        txt_to_save = final_txt["formatted_response"]
    add_message(session, user_id, "assistant", str(txt_to_save))

    return final_txt
