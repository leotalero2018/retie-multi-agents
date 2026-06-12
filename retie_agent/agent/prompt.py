from typing import List, Dict, Optional

# System prompt compartido por los nodos de respuesta del grafo.
SYSTEM_PROMPT_RETIE = (
    "Eres un asistente experto en el Reglamento Técnico de Instalaciones "
    "Eléctricas (RETIE) de Colombia. Respondes en español, con precisión "
    "técnica y solo con base en la evidencia que se te entrega. Cuando la "
    "evidencia incluye valores numéricos, tablas o requisitos normativos, los "
    "reproduces fielmente sin redondear ni omitir."
)

# Reescritura de preguntas de seguimiento: convierte una pregunta que depende
# del historial ("¿y eso aplica en baja tensión?") en una autocontenida para
# que el retrieval (Chroma/NotebookLM) reciba todo el contexto necesario.
CONDENSE_PROMPT = (
    "Dada la conversación previa y la última pregunta del usuario, reescribe la "
    "última pregunta como una pregunta AUTOCONTENIDA sobre el RETIE, en español, "
    "que pueda entenderse sin leer la conversación.\n"
    "Reglas:\n"
    "- Si la pregunta ya es autocontenida, devuélvela EXACTAMENTE igual.\n"
    "- No respondas la pregunta; solo reescríbela.\n"
    "- Conserva términos técnicos, números de artículo, tablas y unidades.\n"
    "- Devuelve únicamente la pregunta reescrita, sin comillas ni explicación.\n\n"
    "CONVERSACIÓN PREVIA:\n{history}\n\n"
    "ÚLTIMA PREGUNTA: {question}\n\n"
    "Pregunta reescrita:"
)


def format_history_for_condense(history: List[Dict[str, str]], max_messages: int = 6) -> str:
    """Formatea los últimos mensajes para el prompt de condensación."""
    lines: List[str] = []
    for msg in history[-max_messages:]:
        role = "Usuario" if msg.get("role") == "user" else "Asistente"
        content = (msg.get("content") or "").strip()
        if len(content) > 500:
            content = content[:500] + "…"
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(sin historial)"


# Two templates:
# - USER: no citations requested
# - ADMIN: asks the model to cite [archivo, página]

PROMPT_TEMPLATE_USER = (
    "Eres un asistente experto en el Reglamento Técnico de Instalaciones Eléctricas (RETIE). "
    "Tu función es responder de forma clara y precisa usando EXCLUSIVAMENTE la información del CONTEXTO. "
    "No inventes ni supongas datos que no estén en el CONTEXTO.\n\n"
    "REGLAS IMPORTANTES:\n"
    "- Si la información está en el CONTEXTO, respóndela completa con TODOS los datos disponibles.\n"
    "- Si el CONTEXTO contiene tablas, listas o valores numéricos, REPRODÚCELOS LITERALMENTE sin omitir filas.\n"
    "- Solo di que la información no está disponible si REALMENTE no aparece en el CONTEXTO.\n"
    "- No cites archivos ni páginas. Usa un tono profesional y estructurado.\n\n"
    "CONTEXTO:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Responde en español con todos los datos del CONTEXTO relevantes para la pregunta."
)


PROMPT_TEMPLATE_ADMIN = (
    "Eres un asistente experto en el Reglamento Técnico de Instalaciones Eléctricas (RETIE). "
    "Responde únicamente con base en la información provista en CONTEXTO. "
    "No inventes datos ni interpretes fuera del contenido disponible. "
    "Si la respuesta no aparece de forma literal, elabora un resumen claro y fiel a la evidencia.\n\n"
    "Cita cada idea relevante en formato [archivo, página], cuando esta información esté disponible en el CONTEXTO. "
    "Evita decir frases como 'No tengo evidencia...' si existen datos relacionados. "
    "En caso de que la información sea ambigua o incompleta, acláralo de forma educada y ofrece una interpretación razonable basada en el texto.\n\n"
    "Usa un tono profesional, preciso y estructurado. Responde en español, de manera concisa, coherente y centrada en el contenido técnico del RETIE.\n\n"
    "CONTEXTO:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Responde en español, concisa y con citas en formato [archivo, página] cuando sea posible."
)




def build_context(blocks: List[Dict], *, include_meta: bool) -> str:
    """
    If include_meta is False (normal users), DO NOT reveal source names/pages.
    If include_meta is True (admins), include [i] source pX headers.
    """
    lines: List[str] = []
    for i, b in enumerate(blocks, 1):
        text = b["text"]
        if include_meta:
            meta = b.get("meta", {})
            src = meta.get("source", "?")
            page = meta.get("page", "?")
            lines.append(f"[{i}] {src} p{page}:\n\"{text}\"\n")
        else:
            # User context: content only, no source metadata
            lines.append(f"\"{text}\"\n")
    return "\n".join(lines)


def make_prompt(context_blocks: List[Dict], question: str, *, is_admin: bool = False) -> str:
    ctx = build_context(context_blocks, include_meta=is_admin)
    tmpl = PROMPT_TEMPLATE_ADMIN if is_admin else PROMPT_TEMPLATE_USER
    return tmpl.format(context=ctx, question=question)


PROMPT_TEMPLATE_HYBRID = (
    "Eres un asistente experto en el Reglamento Técnico de Instalaciones Eléctricas (RETIE). "
    "Dispones de dos fuentes complementarias: fragmentos del documento oficial y un análisis "
    "previo de NotebookLM. Sintetiza ambas fuentes para dar una respuesta precisa y en español. "
    "No inventes información ni contradigas lo que dicen las fuentes. "
    "Si hay contradicción entre fuentes, menciona ambas versiones.\n\n"
    "REGLA IMPORTANTE: Si alguna fuente contiene tablas, listas o valores numéricos, "
    "REPRODÚCELOS LITERALMENTE sin omitir filas ni resumir los datos.\n\n"
    "FRAGMENTOS DEL DOCUMENTO RETIE:\n{context}\n\n"
    "ANÁLISIS COMPLEMENTARIO (NotebookLM):\n{nlm_answer}\n\n"
    "Pregunta: {question}\n"
    "Responde en español con todos los datos relevantes de ambas fuentes."
)


def make_hybrid_prompt(
    context_blocks: List[Dict],
    nlm_answer: str,
    question: str,
    *,
    is_admin: bool = False,
) -> str:
    """Prompt that merges Chroma chunks with a NotebookLM pre-synthesized answer."""
    has_chroma = bool(context_blocks)
    has_nlm = bool(nlm_answer and nlm_answer.strip())

    if has_chroma and has_nlm:
        ctx = build_context(context_blocks, include_meta=is_admin)
        return PROMPT_TEMPLATE_HYBRID.format(
            context=ctx, nlm_answer=nlm_answer.strip(), question=question
        )
    if has_chroma:
        return make_prompt(context_blocks, question, is_admin=is_admin)
    # NLM answer only
    return (
        "Eres un experto en RETIE. Responde con base en el siguiente análisis:\n\n"
        f"{nlm_answer.strip()}\n\n"
        f"Pregunta: {question}\n"
        "Responde en español, de manera breve y precisa."
    )
