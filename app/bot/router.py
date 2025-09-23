# app/bot/router.py
import os
import re
import asyncio
from typing import Dict, Set, Optional, List

from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

from app.config import settings
from app.agent.registry import AGENTS
from app.agent.graph import run_graph
from app.retriever.chroma_client import get_collection

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
        # Keep reply generic (don’t leak policy)
        asyncio.create_task(message.answer("⛔ Comando solo para administradores."))
        return False
    return True

# -----------------------------------------------------------------------------
# Chat state
# -----------------------------------------------------------------------------
CHAT_AGENT: Dict[int, str] = {}
DEFAULT_AGENT = "plumber"
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
            (?:\s*,\s*(?:p|pp)\.?\s*\d+)?
            (?:\s*,\s*\d{1,3})*
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
    parts = (message.text or "").strip().split()
    if len(parts) < 2:
        return await message.answer("Formato: /agent <key>")
    key = parts[1].lower()
    if key not in AGENTS:
        return await message.answer("Agente inválido.")
    CHAT_AGENT[message.chat.id] = key
    cfg = AGENTS[key]
    await message.answer(
        f"✅ Agente: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})"
    )

@router.message(Command("who"))
async def on_who(message: Message):
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    cfg = AGENTS[key]
    await message.answer(
        f"Agente actual: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})"
    )

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
@router.message(Command("debug"))
async def on_debug(message: Message):
    if not _require_admin(message):
        return
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    cfg = AGENTS[key]
    try:
        col = get_collection(cfg.collection)
        metadatas = col.get().get("metadatas", [])
        pdfs = sorted({m.get("source", "desconocido") for m in metadatas})
        total_chunks = len(metadatas)
    except Exception as e:
        pdfs = []
        total_chunks = f"error: {e}"
    await message.answer(
        f"CHROMA_DB_DIR={settings.CHROMA_DB_DIR}\n"
        f"Agente={key}\nColección={cfg.collection}\n"
        f"Total chunks={total_chunks}\n"
        f"PDFs indexados:\n- " + ("\n- ".join(pdfs) if pdfs else "(ninguno)")
    )

@router.message(Command("docs"))
async def on_docs(message: Message):
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
    cfg = AGENTS[key]
    try:
        col = get_collection(cfg.collection)
        result = col.query(query_texts=[last_q], n_results=k)
        metadatas_lists: List[List[dict]] = result.get("metadatas", [[]])
        mlist = metadatas_lists[0] if metadatas_lists else []

        seen = set()
        lines = []
        for i, m in enumerate(mlist, start=1):
            src = m.get("source", "desconocido")
            page = m.get("page", m.get("page_number", ""))
            title = m.get("title", os.path.basename(src) if isinstance(src, str) else "desconocido")
            key_uniq = (src, page, title)
            if key_uniq in seen:
                continue
            seen.add(key_uniq)
            lines.append(f"{i}. {title}  (src={src}, page={page})")

        if not lines:
            return await message.answer("No se encontraron documentos para la última consulta.")

        await message.answer("📄 Documentos más relevantes para la última consulta:\n" + "\n".join(lines))
    except Exception as e:
        await message.answer(f"⚠️ No se pudieron recuperar documentos: {e}")

# -----------------------------------------------------------------------------
# Catch-all text (user questions)
# -----------------------------------------------------------------------------
@router.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    user_id = str(message.from_user.id) if message.from_user else None
    session_id = f"tg:{message.chat.id}"

    LAST_QUERY[message.chat.id] = q

    # typing action through the message's bot (no global Bot needed)
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    loop = asyncio.get_running_loop()
    raw_resp = await loop.run_in_executor(None, run_graph, q, user_id, session_id, agent_key)

    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    resp = _clean_for_user(raw_resp, is_admin)

    await message.answer(resp)
