from typing import List, Dict
from app.config import settings

# ========= OPENAI =========
def _openai_chat(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=settings.MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


# ========= EXTRACTIVO (sin clave) =========
#verificar el envío de respuestas, son muy tomadas del texto, es decir no hay 
#una respuesta sin demasiado contexto, muy general actualmente
def _extractive_answer(blocks: List[Dict], question: str) -> str:
    """
    Toma los bloques (hits) y arma una respuesta corta usando solo su texto.
    Reglas:
    - Si no hay hits: "No tengo evidencia en los documentos."
    - Si hay, devuelve un resumen conciso + citas [archivo, p].
    """
    if not blocks:
        return "No tengo evidencia en los documentos."

    # Concatenamos 1–2 frases por bloque, truncando
    snippets = []
    for b in blocks:
        txt = b["text"].strip().replace("\n", " ")
        if len(txt) > 350:
            txt = txt[:350] + "..."
        src = b["meta"].get("source", "?")
        page = b["meta"].get("page", 0)
        snippets.append(f"- {txt} [{src}, p{page}]")

    # Respuesta básica (extractiva)
    header = "Con base en los documentos, encontré lo siguiente:"
    body = "\n".join(snippets)
    return f"{header}\n{body}"


def chat_answer(prompt: str, blocks: List[Dict], question: str) -> str:
    provider = getattr(settings, "CHAT_PROVIDER", "openai")
    if provider == "openai":
        return _openai_chat(prompt)
    else:
        # "extractive" u otro → respuesta sin LLM
        return _extractive_answer(blocks, question)
