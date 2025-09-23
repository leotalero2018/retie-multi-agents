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


# ========= EXTRACTIVE (sin clave) =========
def _extractive_answer(blocks: List[Dict], question: str, *, is_admin: bool) -> str:
    """
    For admins, include citations. For normal users, no citations or source metadata.
    """
    if not blocks:
        return "No tengo evidencia en los documentos."

    snippets = []
    for b in blocks:
        txt = b["text"].strip().replace("\n", " ")
        if len(txt) > 350:
            txt = txt[:350] + "..."
        if is_admin:
            src = b.get("meta", {}).get("source", "?")
            page = b.get("meta", {}).get("page", 0)
            snippets.append(f"- {txt} [{src}, p{page}]")
        else:
            snippets.append(f"- {txt}")

    header = "Con base en los documentos, encontré lo siguiente:"
    body = "\n".join(snippets)
    return f"{header}\n{body}"


def chat_answer(prompt: str, blocks: List[Dict], question: str, *, is_admin: bool = False) -> str:
    provider = getattr(settings, "CHAT_PROVIDER", "openai")
    if provider == "openai":
        # The prompt itself is already admin-aware (will/won’t ask for citations).
        return _openai_chat(prompt)
    else:
        # “extractive” or other fallback: also admin-aware for citations
        return _extractive_answer(blocks, question, is_admin=is_admin)
