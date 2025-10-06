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
from app.agent.registry import AGENTS as _AGENTS  # optional registry

# Mongo GridFS saver (make sure MONGO_URI, MONGO_DB, MONGO_BUCKET are set)
from app.services.mongo_store import save_image_from_path

# --- create router FIRST (before any @router.message decorators) ---
router = Router(name="telegram_router")

# --- single agent instance ---
_agent = RetieAgent()

# --- admin controls ---
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

# --- chat state ---
CHAT_AGENT: Dict[int, str] = {}
DEFAULT_AGENT = "plumber"
LAST_QUERY: Dict[int, str] = {}

# --- output cleaning for non-admins ---
_SOURCES_HEADERS = (
    r"^\s*fuentes\s*:\s*$", r"^\s*referencias\s*:\s*$",
    r"^\s*sources\s*:\s*$", r"^\s*citas\s*:\s*$"
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

# --- observability (Langfuse v3 wrappers) ---
from app.observability.obs import trace_ctx, span_ctx, log_generation
from app.config import settings

# --- voice transcription ---
from app.services.whisper import transcribe_audio

# --- image OCR / vision (question-first path) ---
from app.services.vision import extract_question_from_image, vision_extract_insights


# ----------------------- commands & admin -----------------------

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
    try:
        from app.retriever.retrieve import search
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


# ----------------------- helpers -----------------------

async def _download_to_tmp(bot, file_id: str, suffix: str) -> Path:
    """Download a Telegram file into a temp path and return the path."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tf.name)
    tf.close()
    tg_file = await bot.get_file(file_id)
    await bot.download_file(tg_file.file_path, destination=tmp_path)
    return tmp_path

def _to_wav_if_needed(src: Path) -> Path:
    """Normalize to 16kHz mono WAV using ffmpeg; fallback to original on error."""
    try:
        import ffmpeg
        out = Path(tempfile.mkstemp(suffix=".wav")[1])
        (
            ffmpeg
            .input(str(src))
            .filter("loudnorm", i=-16, tp=-1.5, lra=11)
            .output(str(out), format="wav", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=True)
        )
        return out
    except Exception:
        return src

async def _download_image_best(bot, photo_sizes) -> Path:
    """Download the largest Telegram photo into temp and return path."""
    biggest = max(photo_sizes, key=lambda p: p.file_size or 0)
    tg_file = await bot.get_file(biggest.file_id)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp = Path(tf.name)
    tf.close()
    await bot.download_file(tg_file.file_path, destination=tmp)
    return tmp

async def _download_image_document(bot, document) -> Path:
    """Download an image sent as document into temp and return path."""
    suffix = Path(document.file_name or "image.jpg").suffix or ".jpg"
    tg_file = await bot.get_file(document.file_id)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp = Path(tf.name)
    tf.close()
    await bot.download_file(tg_file.file_path, destination=tmp)
    return tmp

async def _handle_image_common(message: Message, img_path: Path):
    """Shared pipeline: save → read question → RAG answer."""
    # 0) Persist to Mongo (best-effort)
    mongo_id = None
    try:
        mongo_id = save_image_from_path(
            img_path,
            filename=f"tg_{message.chat.id}_{message.message_id}{img_path.suffix or '.jpg'}",
            content_type="image/jpeg",
            metadata={
                "chat_id": message.chat.id,
                "user_id": message.from_user.id if message.from_user else None,
                "caption": message.caption or "",
                "source": "telegram",
            },
        )
    except Exception:
        mongo_id = None

    # 1) Try to read the user's question from the image (OCR → Vision fallback)
    recognized_q = ""
    try:
        recognized_q = extract_question_from_image(img_path)
    except Exception:
        recognized_q = ""

    if not recognized_q:
        recognized_q = (message.caption or "").strip()
        if not recognized_q:
            try:
                recognized_q = vision_extract_insights(img_path) or ""
            except Exception:
                recognized_q = ""

    if not recognized_q:
        return await message.answer(
            "No pude leer claramente la pregunta de la imagen. "
            "Prueba con una foto más nítida o envíala con mayor resolución."
        )

    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    LAST_QUERY[message.chat.id] = recognized_q
    is_admin = _is_admin(message.from_user.id if message.from_user else None)

    # Optional: show what was read & GridFS link (admins only)
    if is_admin:
        text = f"🖼️ Leí esta pregunta:\n“{recognized_q}”"
        if mongo_id:
            text += f"\n💾 Guardada: /files/{mongo_id}"
        await message.answer(text)

    user_id = str(message.from_user.id) if message.from_user else None
    meta = {"agent_key": agent_key, "chat_id": message.chat.id, "via": "image"}

    with trace_ctx("telegram.image", user_id=user_id, metadata=meta) as tr:
        with span_ctx(tr, "agent.answer", metadata={"question": recognized_q}):
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: _agent.answer(recognized_q, agent_key=agent_key, is_admin=is_admin),
            )
        try:
            log_generation(
                tr, name="openai.chat",
                input_text=recognized_q,
                output_text=raw,
                model=getattr(settings, "CHAT_MODEL", "gpt-4o-mini"),
                usage={}, metadata={"via": "image", "agent_key": agent_key},
            )
        except Exception:
            pass

    resp = _clean_for_user(raw, is_admin)
    await message.answer(resp)


# ----------------------- VOICE/AUDIO -----------------------

@router.message(F.voice | F.audio)
async def on_voice(message: Message):
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    tg_obj = message.voice or message.audio
    suffix = ".oga" if message.voice else (Path((tg_obj.file_name or "audio.mp3")).suffix or ".mp3")

    local_file = await _download_to_tmp(message.bot, tg_obj.file_id, suffix)

    wav_file: Optional[Path] = None
    try:
        wav_file = _to_wav_if_needed(local_file)
        transcript = transcribe_audio(wav_file, language="es").strip()
    except Exception as e:
        try:
            local_file.unlink(missing_ok=True)
            if wav_file and wav_file != local_file:
                wav_file.unlink(missing_ok=True)
        finally:
            return await message.answer(f"⚠️ Error al transcribir el audio: {e}")

    try:
        local_file.unlink(missing_ok=True)
        if wav_file and wav_file != local_file:
            wav_file.unlink(missing_ok=True)
    except Exception:
        pass

    if not transcript:
        return await message.answer("No pude entender el audio 😕")

    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    LAST_QUERY[message.chat.id] = transcript
    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    meta = {"agent_key": agent_key, "chat_id": message.chat.id, "via": "voice"}

    with trace_ctx("telegram.voice", user_id=str(message.from_user.id) if message.from_user else None, metadata=meta) as tr:
        with span_ctx(tr, "agent.answer", metadata={"question": transcript}):
            loop = asyncio.get_running_loop()
            raw_resp = await loop.run_in_executor(
                None,
                lambda: _agent.answer(transcript, agent_key=agent_key, is_admin=is_admin),
            )
        try:
            log_generation(
                tr,
                name="openai.chat",
                input_text=transcript,
                output_text=raw_resp,
                model=getattr(settings, "CHAT_MODEL", "gpt-4o-mini"),
                usage={},
                metadata={"agent_key": agent_key, "via": "voice"},
            )
        except Exception:
            pass

    resp = _clean_for_user(raw_resp, is_admin)
    await message.answer(resp)


# ----------------------- IMAGES -----------------------

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

@router.message(F.document)
async def on_image_document(message: Message):
    # Only handle if it's an image/*
    if not (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")):
        return  # ignore other documents
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    img_path = await _download_image_document(message.bot, message.document)
    try:
        await _handle_image_common(message, img_path)
    finally:
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass


# ----------------------- TEXT -----------------------

@router.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    LAST_QUERY[message.chat.id] = q

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    user_id = str(message.from_user.id) if message.from_user else None
    meta = {"agent_key": agent_key, "chat_id": message.chat.id}

    with trace_ctx("telegram.message", user_id=user_id, metadata=meta) as tr:
        with span_ctx(tr, "agent.answer", metadata={"question": q}):
            loop = asyncio.get_running_loop()
            raw_resp = await loop.run_in_executor(
                None,
                lambda: _agent.answer(
                    q,
                    agent_key=agent_key,
                    is_admin=_is_admin(message.from_user.id if message.from_user else None),
                ),
            )
        try:
            log_generation(
                tr,
                name="openai.chat",
                input_text=q,
                output_text=raw_resp,
                model=getattr(settings, "CHAT_MODEL", "gpt-4o-mini"),
                usage={},
                metadata={"agent_key": agent_key},
            )
        except Exception:
            pass

    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    resp = _clean_for_user(raw_resp, is_admin)
    await message.answer(resp)
