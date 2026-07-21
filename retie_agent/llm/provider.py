# app/llm/provider.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import re
import logging
import httpx

from retie_agent.config import settings

logger = logging.getLogger(__name__)

try:  # el símbolo existe en cualquier SDK reciente; fallback defensivo por si acaso
    from openai import BadRequestError
except Exception:  # pragma: no cover
    BadRequestError = Exception  # type: ignore


# Parámetros que cada modelo ya demostró NO soportar → se omiten en llamadas
# futuras con ese mismo modelo para no repetir el 400 (cache por proceso).
_MODEL_UNSUPPORTED: Dict[str, set] = {}


def _unsupported_param_from_error(err: Exception) -> Optional[str]:
    """Extrae el nombre del parámetro que la API rechazó en un 400.

    El cuerpo del error trae {'param': 'max_tokens'/'temperature'/...}; se lee de
    ahí y, como respaldo, por regex sobre el mensaje ('Unsupported parameter/value').
    """
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            p = inner.get("param")
            if isinstance(p, str) and p:
                return p
    msg = str(err)
    m = re.search(r"'param':\s*'([^']+)'", msg)
    if m:
        return m.group(1)
    m = re.search(r"[Uu]nsupported (?:parameter|value): '([^']+)'", msg)
    return m.group(1) if m else None


def create_chat_completion(client, **kwargs):
    """Wrapper de client.chat.completions.create compatible entre familias de modelos.

    1. Traduce ``max_tokens`` → ``max_completion_tokens``: los modelos GPT-5 y
       o-series rechazan ``max_tokens`` (HTTP 400 unsupported_parameter), y
       ``max_completion_tokens`` también lo aceptan los modelos previos (gpt-4o*),
       así que el reemplazo es universal.
    2. Auto-recuperación: si el modelo rechaza otro parámetro (p. ej. ``temperature``
       distinta de 1 en GPT-5), lo omite y reintenta, recordándolo para no repetir
       el 400 en las siguientes llamadas con el mismo modelo.

    Úsalo en lugar de ``client.chat.completions.create(...)`` en todo el código.
    """
    if "max_tokens" in kwargs:
        # No pisar un max_completion_tokens explícito si por alguna razón ya viniera.
        kwargs.setdefault("max_completion_tokens", kwargs["max_tokens"])
        kwargs.pop("max_tokens", None)

    model = kwargs.get("model", "") or ""
    for param in list(_MODEL_UNSUPPORTED.get(model, ())):
        kwargs.pop(param, None)

    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as err:
            param = _unsupported_param_from_error(err)
            # Solo reintentamos si el parámetro ofensor está presente y es removible.
            if not param or param not in kwargs:
                raise
            kwargs.pop(param, None)
            _MODEL_UNSUPPORTED.setdefault(model, set()).add(param)
            logger.warning(
                "Modelo %r no soporta el parámetro %r — se omite y se reintenta.",
                model, param,
            )

# --------------------------- OpenAI client --------------------------- #
def _build_openai_client():
    from openai import OpenAI

    # Optional proxy + custom base URL (Azure/OpenRouter/etc.)
    # httpx 0.28 eliminó el kwarg `proxies`; el singular `proxy` existe desde 0.26.
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
    http_client = httpx.Client(proxy=proxy, timeout=60) if proxy else None
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
    mdl = model or getattr(settings, "CHAT_MODEL", "gpt-4.1")
    maxt = max_tokens or getattr(settings, "MAX_TOKENS", 512)
    resp = create_chat_completion(
        _client(),
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


