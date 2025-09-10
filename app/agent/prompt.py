# Se le asigna un prompt dependiendo de los agentes a crear, cada prompt se le asigna una funcionalidad


from typing import List, Dict

PROMPT_TEMPLATE = (
    "Eres un experto en RETIE. Usa únicamente la información provista en CONTEXTO.\n"
    "Si no hay información suficiente, responde: 'No tengo evidencia en los documentos.'\n\n"
    "CONTEXTO:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Responde en español, conciso y con citas en formato [archivo, p\u00e1gina].\n"
)


def build_context(blocks: List[Dict]) -> str:
    lines = []
    for i, b in enumerate(blocks, 1):
        meta = b["meta"]
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        text = b["text"]
        lines.append(f"[{i}] {src} p{page}:\n\"{text}\"\n")
    return "\n".join(lines)


def make_prompt(context_blocks: List[Dict], question: str) -> str:
    ctx = build_context(context_blocks)
    return PROMPT_TEMPLATE.format(context=ctx, question=question)