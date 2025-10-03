# app/llm/provider.py
from __future__ import annotations

from typing import List, Dict, Any
import os
import httpx

from app.config import settings

# --------------------------- OpenAI client --------------------------- #
def _build_openai_client():
    from openai import OpenAI

    # Optional proxy + custom base URL (Azure/OpenRouter/etc.)
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
    http_client = httpx.Client(proxies=proxy, timeout=60) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None

    api_key = (
        os.getenv("OPENAI_API_KEY")           # prefer env directly
        or getattr(settings, "OPENAI_API_KEY", None)
    )
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or settings.")

    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# Reusable singleton
_openai_client = None
def _client():
    global _openai_client
    if _openai_client is None:
        _openai_client = _build_openai_client()
    return _openai_client


# --------------------------- Public helpers --------------------------- #
def chat(messages: List[Dict[str, str]], *, model: str | None = None, max_tokens: int | None = None, temperature: float = 0.0) -> str:
    """
    Minimal role-based chat helper.
    messages = [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    """
    mdl = model or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    maxt = max_tokens or getattr(settings, "MAX_TOKENS", 512)
    resp = _client().chat.completions.create(
        model=mdl,
        temperature=temperature,
        messages=messages,
        max_tokens=maxt,
    )
    return (resp.choices[0].message.content or "").strip()


def _extractive_answer(blocks: List[Dict[str, Any]], question: str, *, is_admin: bool) -> str:
    """
    Simple non-LLM fallback that summarizes retrieved blocks.
    Admins get citations; users do not.
    """
    if not blocks:
        return "No tengo evidencia en los documentos."

    snippets = []
    for b in blocks:
        txt = (b.get("text") or "").strip().replace("\n", " ")
        if len(txt) > 350:
            txt = txt[:350] + "..."
        if is_admin:
            meta = b.get("meta", {})
            src = meta.get("source", "?")
            page = meta.get("page", 0)
            snippets.append(f"- {txt} [{src}, p{page}]")
        else:
            snippets.append(f"- {txt}")

    header = "Con base en los documentos, encontré lo siguiente:"
    return f"{header}\n" + "\n".join(snippets)

def chat_answer(prompt: str, blocks: List[Dict[str, Any]], question: str, *, is_admin: bool = False) -> str:
    """
    - Si CHAT_PROVIDER=openai: llama al LLM con un mensaje de sistema sensible al rol.
    - Si HAY bloques recuperados pero el LLM aún devuelve la frase 'no evidencia',
      hacer fallback a un resumen extractivo (admin mantiene citas; usuario no).
    - Si provider != openai: usar extractivo directamente.
    """
    provider = getattr(settings, "CHAT_PROVIDER", "openai").lower()

    def _looks_like_no_evidence(text: str) -> bool:
        t = (text or "").strip().lower()
        return "no tengo evidencia en los documentos" in t

    if provider == "openai":
        sys_rules = "Eres un asistente útil."
        if blocks:
            # Regla dura: cuando hay evidencia, nunca dar la frase 'no evidencia'.
            if is_admin:
                sys_rules += (
                    " Tienes evidencia en el CONTEXTO. Si no hay una definición literal, "
                    "redacta el mejor resumen posible usando el CONTEXTO y cita en formato "
                    "[archivo, página]. Evita responder 'No tengo evidencia en los documentos.'"
                )
            else:
                sys_rules += (
                    " Tienes evidencia en el CONTEXTO. Si no hay una definición literal, "
                    "redacta el mejor resumen posible usando el CONTEXTO sin citar archivos "
                    "ni páginas. Evita responder 'No tengo evidencia en los documentos.'"
                )

        msgs = [
            {"role": "system", "content": sys_rules},
            {"role": "user", "content": prompt},
        ]
        out = chat(msgs)

        # Red de seguridad para AMBOS roles: si hay hits pero el modelo se niega, usar extractivo
        if blocks and _looks_like_no_evidence(out):
            return _extractive_answer(blocks, question, is_admin=is_admin)
        return out

    # Ruta no-OpenAI
    return _extractive_answer(blocks, question, is_admin=is_admin)


