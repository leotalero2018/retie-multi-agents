# app/bot/router.py
from __future__ import annotations

import os
import re
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Set, Optional, List

from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

from app.agent.retie_agent import RetieAgent
from app.agent.registry import AGENTS as _AGENTS
from app.services.mongo_store import save_image_from_path

from app.services.vision import (
    extract_question_from_image,
    vision_extract_insights,
    ocr_image,
    analyze_question_image,
    is_question_image,
)

from app.observability.obs import trace_ctx, span_ctx, log_generation
from app.config import settings
from app.services.whisper import transcribe_audio

router = Router(name="telegram_router")
_agent = RetieAgent()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
_raw_ids = os.getenv("ADMIN_USER_IDS", "").strip()
STATIC_ADMIN_IDS: Set[int] = {int(x) for x in _raw_ids.split(",") if x.strip().isdigit()} if _raw_ids else set()
RUNTIME_ADMINS: Set[int] = set()

CHAT_AGENT: Dict[int, str] = {}
DEFAULT_AGENT = "plumber"
LAST_QUERY: Dict[int, str] = {}


def _is_admin(uid: Optional[int]) -> bool:
    return bool(uid and (uid in STATIC_ADMIN_IDS or uid in RUNTIME_ADMINS))


# --- helper cleaning for users ---
def _clean_for_user(raw: str, is_admin: bool) -> str:
    if is_admin:
        return raw
    text = re.sub(r"\[.*?\]", "", raw)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------- Commands ----------------
@router.message(CommandStart())
async def on_start(message: Message):
    CHAT_AGENT[message.chat.id] = DEFAULT_AGENT
    await message.answer("¡Hola! Soy tu bot RETIE, ¿en qué puedo ayudarte?")


@router.message(Command("agent"))
async def on_agent(message: Message):
    parts = (message.text or "").split()
    if len(parts) < 2:
        return await message.answer("Formato: /agent <nombre>")
    key = parts[1].lower()
    CHAT_AGENT[message.chat.id] = key
    await message.answer(f"✅ Agente establecido: {key}")


# ---------------- Image Handling ----------------
async def _download_image_best(bot, photo_sizes) -> Path:
    biggest = max(photo_sizes, key=lambda p: p.file_size or 0)
    tg_file = await bot.get_file(biggest.file_id)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp = Path(tf.name)
    tf.close()
    await bot.download_file(tg_file.file_path, destination=tmp)
    return tmp


async def _handle_image_common(message: Message, img_path: Path):
    """Main image handler: saves, OCRs, and decides flow."""
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)

    # 1️⃣ Persist image (optional)
    try:
        save_image_from_path(
            img_path,
            filename=f"tg_{message.chat.id}_{message.message_id}.jpg",
            content_type="image/jpeg",
            metadata={"chat_id": message.chat.id, "caption": message.caption or "", "source": "telegram"},
        )
    except Exception:
        pass

    # 2️⃣ OCR preview
    txt_raw = ocr_image(img_path)
    if is_question_image(txt_raw):
        result = analyze_question_image(img_path, agent_key)
        if "error" in result:
            return await message.answer(f"⚠️ {result['error']}")
        verdict = "✅ Correcto" if result["is_correct"] else "❌ Incorrecto"
        feedback = (
            f"{verdict}\n\n"
            f"<b>Pregunta:</b> {result['question']}\n"
            f"<b>Tu respuesta:</b> {result['user_answer']}\n"
            f"<b>Respuesta esperada:</b> {result['expected_answer']}\n\n"
            f"{result['explanation']}"
        )
        return await message.answer(feedback)

    # 3️⃣ Generic vision fallback
    recognized_q = extract_question_from_image(img_path) or (message.caption or "").strip()
    if not recognized_q:
        recognized_q = vision_extract_insights(img_path)

    if not recognized_q:
        return await message.answer("No pude leer la pregunta ni el contenido de la imagen 😕")

    LAST_QUERY[message.chat.id] = recognized_q
    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    user_id = str(message.from_user.id) if message.from_user else None
    meta = {"agent_key": agent_key, "chat_id": message.chat.id, "via": "image"}

    with trace_ctx("telegram.image", user_id=user_id, metadata=meta) as tr:
        with span_ctx(tr, "agent.answer", metadata={"question": recognized_q}):
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, lambda: _agent.answer(recognized_q, agent_key=agent_key, is_admin=is_admin))
        try:
            log_generation(tr, "openai.chat", recognized_q, raw, getattr(settings, "CHAT_MODEL", "gpt-4o-mini"), {}, meta)
        except Exception:
            pass

    await message.answer(_clean_for_user(raw, is_admin))


@router.message(F.photo)
async def on_photo(message: Message):
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    img_path = await _download_image_best(message.bot, message.photo)
    try:
        await _handle_image_common(message, img_path)
    finally:
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------- Text ----------------
@router.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    LAST_QUERY[message.chat.id] = q
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    user_id = str(message.from_user.id) if message.from_user else None
    meta = {"agent_key": agent_key, "chat_id": message.chat.id}

    with trace_ctx("telegram.message", user_id=user_id, metadata=meta) as tr:
        with span_ctx(tr, "agent.answer", metadata={"question": q}):
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None, lambda: _agent.answer(q, agent_key=agent_key, is_admin=is_admin)
            )
        try:
            log_generation(tr, "openai.chat", q, raw, getattr(settings, "CHAT_MODEL", "gpt-4o-mini"), {}, meta)
        except Exception:
            pass

    await message.answer(_clean_for_user(raw, is_admin))
