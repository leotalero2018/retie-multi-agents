from typing import List, Dict

# Two templates:
# - USER: no citations requested
# - ADMIN: asks the model to cite [archivo, página]

PROMPT_TEMPLATE_USER = (
    "Eres un asistente experto en el Reglamento Técnico de Instalaciones Eléctricas (RETIE). "
    "Tu función es responder de forma clara, precisa y exclusivamente con base en la información "
    "contenida en CONTEXTO. No inventes información ni hagas suposiciones. "
    "Si la respuesta no se encuentra explícitamente, elabora un resumen razonado usando solo lo disponible. "
    "Indica educadamente si la información no aparece en el CONTEXTO.\n\n"
    "Usa un tono profesional, amable y fácil de entender. Presenta tus respuestas en español, "
    "de forma breve, estructurada y orientada a la comprensión del usuario.\n\n"
    "Evita citar archivos o páginas. No repitas el texto del CONTEXTO, resume y explica con tus propias palabras.\n\n"
    "CONTEXTO:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Responde en español, de manera breve, precisa y sin inventar información."
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
