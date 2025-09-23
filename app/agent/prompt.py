from typing import List, Dict

# Two templates:
# - USER: no citations requested
# - ADMIN: asks the model to cite [archivo, página]
PROMPT_TEMPLATE_USER = (
    "Eres un experto en RETIE. Usa únicamente la información provista en CONTEXTO.\n"
    "Si no hay información suficiente, responde: 'No tengo evidencia en los documentos.'\n\n"
    "CONTEXTO:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Responde en español, de forma breve y clara."
)

PROMPT_TEMPLATE_ADMIN = (
    "Eres un experto en RETIE. Usa únicamente la información provista en CONTEXTO.\n"
    "Si no hay información suficiente, responde: 'No tengo evidencia en los documentos.'\n\n"
    "CONTEXTO:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Responde en español, conciso y con citas en formato [archivo, página]."
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
