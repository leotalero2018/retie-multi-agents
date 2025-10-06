# app/bot/router.py
from __future__ import annotations

import os
import re
import asyncio
from typing import Dict, Set, Optional, List

from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

from app.agent.retie_agent import RetieAgent
from app.agent.registry import AGENTS as _AGENTS  # optional registry (may be empty)

# 🔎 Observability (safe no-op if disabled/misconfigured)
from app.observability.obs import trace_ctx, span_ctx, log_generation
from app.config import settings

# Single agent instance
_agent = RetieAgent()
router = Router(name="telegram_router")

# -----------------------------------------------------------------------------
# Admin controls
# -----------------------------------------------------------------------------
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")

_raw_ids = os.getenv("ADMIN_USER_IDS", "").strip()
STATIC_ADMIN_IDS: Set[int] = {
    int(x) for x in _raw_ids.split(",") if x.strip().isdigit()
} if _raw_ids else set()

RUNTIME_ADMINS: Set[int] = set()

def _is_admin(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    return (user_id in STATIC_ADMIN_IDS) or (user_id in RUNTIME_ADMINS)

def _require_admin(message: Message) -> bool:
    uid = message.from_user.id if message.from_user else None
    if not _is_admin(uid):
        asyncio.create_task(message.answer("⛔ Comando solo para administradores."))
        return False
    return True

# -----------------------------------------------------------------------------
# Chat state
# -----------------------------------------------------------------------------
CHAT_AGENT: Dict[int, str] = {}
DEFAULT_AGENT = "plumber"  # keeps your previous default
LAST_QUERY: Dict[int, str] = {}

# -----------------------------------------------------------------------------
# Output cleaning for non-admins
# -----------------------------------------------------------------------------
_SOURCES_HEADERS = (
    r"^\s*fuentes\s*:\s*$",
    r"^\s*referencias\s*:\s*$",
    r"^\s*sources\s*:\s*$",
    r"^\s*citas\s*:\s*$",
)
_SOURCES_HEADERS_RE = re.compile("|".join(_SOURCES_HEADERS), re.IGNORECASE | re.MULTILINE)

_INLINE_BRACKET_CITE_RE = re.compile(
    r"""
    \[
        \s*
        (?:
            \d{1,3}
            (?:\s*[-–]\s*\d{1,3})?
            (?:\s*,\s*(?:p|pp)\.?\s*\d+)?      # page refs
            (?:\s*,\s*\d{1,3})*                # extra cites
        )
        \s*
    \]
    """,
    re.VERBOSE,
)

def _strip_sources_sections(text: str) -> str:
    m = _SOURCES_HEADERS_RE.search(text)
    if not m:
        return text
    return text[: m.start()].rstrip()

def _strip_inline_citations(text: str) -> str:
    return _INLINE_BRACKET_CITE_RE.sub("", text)

def _clean_for_user(raw: str, is_admin: bool) -> str:
    if is_admin:
        return raw
    cleaned = _strip_sources_sections(raw)
    cleaned = _strip_inline_citations(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

# -----------------------------------------------------------------------------
# Public commands
# -----------------------------------------------------------------------------
@router.message(CommandStart())
async def on_start(message: Message):
    CHAT_AGENT[message.chat.id] = DEFAULT_AGENT
    await message.answer("¡Hola! Soy tu bot RETIE, ¿en qué puedo ayudarte?")

@router.message(Command("agent"))
async def on_agent(message: Message):
    # /agent <key>
    parts = (message.text or "").strip().split()
    if len(parts) < 2:
        return await message.answer("Formato: /agent <key>")
    key = parts[1].lower()
    # If registry present and key must exist there; otherwise allow free text to route collections
    if _AGENTS and key not in _AGENTS:
        return await message.answer("Agente inválido.")
    CHAT_AGENT[message.chat.id] = key
    if _AGENTS and key in _AGENTS:
        cfg = _AGENTS[key]
        return await message.answer(
            f"✅ Agente: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})"
        )
    await message.answer(f"✅ Agente: {key}")

@router.message(Command("who"))
async def on_who(message: Message):
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    if _AGENTS and key in _AGENTS:
        cfg = _AGENTS[key]
        return await message.answer(
            f"Agente actual: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})"
        )
    await message.answer(f"Agente actual: {key}")

# -----------------------------------------------------------------------------
# Admin authentication
# -----------------------------------------------------------------------------
@router.message(Command("admin"))
async def on_admin_login(message: Message):
    parts = (message.text or "").strip().split()
    if len(parts) < 2:
        return await message.answer("Formato: /admin <contraseña>")
    password = parts[1]
    if password != ADMIN_PASSWORD:
        return await message.answer("❌ Contraseña incorrecta")
    if message.from_user:
        RUNTIME_ADMINS.add(message.from_user.id)
    await message.answer("✅ Acceso administrador concedido en esta sesión.")

@router.message(Command("logout"))
async def on_admin_logout(message: Message):
    if message.from_user and message.from_user.id in RUNTIME_ADMINS:
        RUNTIME_ADMINS.discard(message.from_user.id)
        return await message.answer("👋 Sesión de administrador cerrada.")
    return await message.answer("No hay sesión de administrador activa.")

# -----------------------------------------------------------------------------
# Admin-only commands
# -----------------------------------------------------------------------------
@router.message(Command("docs"))
async def on_docs(message: Message):
    """
    Returns top-k doc headers for the last query, using the same retrieval as the agent.
    """
    if not _require_admin(message):
        return

    parts = (message.text or "").strip().split()
    try:
        k = int(parts[1]) if len(parts) >= 2 else 5
        k = max(1, min(20, k))
    except ValueError:
        k = 5

    last_q = LAST_QUERY.get(message.chat.id)
    if not last_q:
        return await message.answer("No hay una consulta previa en este chat. Envía una pregunta primero.")

    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)

    try:
        from app.retriever.retrieve import search  # same function agent uses
        from app.agent.retie_agent import _resolve_collection, _dedupe_hits

        coll = _resolve_collection(agent_key=key, explicit=None)
        hits = _dedupe_hits(search(last_q, top_k=k, collection_name=coll))

        if not hits:
            return await message.answer("No se encontraron documentos para la última consulta.")

        seen = set()
        lines: List[str] = []
        for i, h in enumerate(hits, start=1):
            meta = h.get("meta", {})
            src = meta.get("source", "desconocido")
            page = meta.get("page", meta.get("page_number", ""))
            title = meta.get("title", os.path.basename(src) if isinstance(src, str) else "desconocido")
            key_uniq = (src, page, title)
            if key_uniq in seen:
                continue
            seen.add(key_uniq)
            lines.append(f"{i}. {title}  (src={src}, page={page})")

        await message.answer("📄 Documentos más relevantes para la última consulta:\n" + "\n".join(lines))
    except Exception as e:
        await message.answer(f"⚠️ No se pudieron recuperar documentos: {e}")

# -----------------------------------------------------------------------------
# Catch-all text (user questions)  — now with Langfuse trace/span/generation
# -----------------------------------------------------------------------------
@router.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    user_id = str(message.from_user.id) if message.from_user else None
    session_id = f"tg:{message.chat.id}"

    LAST_QUERY[message.chat.id] = q

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    is_admin_flag = _is_admin(message.from_user.id if message.from_user else None)

    # End-to-end trace for this message
    with trace_ctx(
        "tg_message",
        user_id=user_id,
        metadata={
            "chat_id": message.chat.id,
            "agent_key": agent_key,
            "admin": is_admin_flag,
        },
    ) as trace:
        with span_ctx(trace, "agent_answer", {"provider": settings.CHAT_PROVIDER, "model": settings.CHAT_MODEL}):
            loop = asyncio.get_running_loop()
            raw_resp = await loop.run_in_executor(
                None,
                lambda: _agent.answer(q, agent_key=agent_key, is_admin=is_admin_flag),
            )

        resp = _clean_for_user(raw_resp, is_admin_flag)

        # Log final generation (the text we actually return to user)
        try:
            log_generation(
                trace,
                name="final_answer",
                input_text=q,
                output_text=resp,
                model=getattr(settings, "CHAT_MODEL", ""),
                usage=None,
                metadata={"admin": is_admin_flag, "agent_key": agent_key},
            )
        except Exception:
            pass

        await message.answer(resp)
