# app/bot/router.py
from __future__ import annotations

import os
import re
import asyncio
import tempfile
import contextvars
from pathlib import Path
from typing import Dict, Set, Optional, List

from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart, Command

# Run the LangGraph pipeline (instrumented for Langfuse)
from retie_agent.agent.graph import run_graph
from retie_agent.agent.registry import AGENTS as _AGENTS
from retie_agent.services.history import clear_history

# --- create router FIRST (before any @router.message decorators) ---
router = Router(name="telegram_router")

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
# "auto" = sin agente fijo → consulta todas las colecciones (norma_vigente + normas_historicas)
DEFAULT_AGENT = "auto"
LAST_QUERY: Dict[int, str] = {}


# --- output cleaning for non-admins ---
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


_PRE_BLOCK_RE = re.compile(r"<pre>.*?</pre>", re.DOTALL | re.IGNORECASE)


def _clean_for_user(raw: str, is_admin: bool) -> str:
    if is_admin:
        return raw

    # Proteger bloques <pre> (tablas ASCII): su alineación depende de espacios
    # múltiples, que el colapso de whitespace de abajo destruiría.
    pre_blocks: List[str] = []

    def _stash(m: "re.Match") -> str:
        pre_blocks.append(m.group(0))
        return f"\x00PRE{len(pre_blocks) - 1}\x00"

    cleaned = _PRE_BLOCK_RE.sub(_stash, raw)
    cleaned = _strip_sources_sections(cleaned)
    cleaned = _strip_inline_citations(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Restaurar los bloques <pre> intactos
    for i, block in enumerate(pre_blocks):
        cleaned = cleaned.replace(f"\x00PRE{i}\x00", block)
    return cleaned


# Telegram HTML mode only supports a small subset of tags.
# Any other tag (e.g. <textarea>, <form>, <div>) causes TelegramBadRequest.
_TELEGRAM_SAFE_TAGS = re.compile(
    r"<(?!/?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|a|tg-spoiler|tg-emoji)\b)[^>]+>",
    re.IGNORECASE,
)


def _sanitize_html_for_telegram(text: str) -> str:
    """Strip HTML tags that Telegram's HTML parser does not support.

    Keeps <b>, <i>, <u>, <s>, <code>, <pre>, <a> and a few others.
    Everything else (e.g. <textarea>, <form>, <div>) is removed so Telegram
    doesn't reject the message with 'can't parse entities'.
    """
    return _TELEGRAM_SAFE_TAGS.sub("", text)


def _finalize_for_user(result, is_admin: bool) -> str:
    """Prepara la respuesta final para el usuario:
    1. Extrae el texto de la respuesta (payload del stylist o texto plano).
    2. Lo limpia según el rol (quita citas crudas del LLM para no-admins).
    3. Anexa las 'Fuentes consultadas' (página y sección) construidas a partir
       de los documentos realmente recuperados — se añaden DESPUÉS de limpiar
       para garantizar que siempre se muestren y den trazabilidad al usuario.
    4. Sanitiza tags HTML no soportados por Telegram (parse_mode=HTML).
    """
    if isinstance(result, dict):
        text = result.get("formatted_response") or "No se obtuvo respuesta del agente."
        sources_text = result.get("sources_text", "")
    else:
        text = str(result or "No se obtuvo respuesta del agente.")
        sources_text = ""

    cleaned = _clean_for_user(text, is_admin)
    if sources_text:
        cleaned = f"{cleaned}\n\n{sources_text}"
    return _sanitize_html_for_telegram(cleaned)


_TELEGRAM_CAPTION_LIMIT = 1024


async def _send_response(message: Message, result, is_admin: bool) -> None:
    """Entrega la respuesta al usuario: como foto si el agente generó una imagen
    de tabla, o como texto en cualquier otro caso."""
    image = result.get("image") if isinstance(result, dict) else None
    if image:
        caption = _finalize_for_user(result, is_admin)
        if len(caption) > _TELEGRAM_CAPTION_LIMIT:
            caption = caption[: _TELEGRAM_CAPTION_LIMIT - 1] + "…"
        photo = BufferedInputFile(image, filename="tabla.png")
        await message.answer_photo(photo, caption=caption or None)
    else:
        await message.answer(_finalize_for_user(result, is_admin))


# --- voice transcription ---
from retie_agent.services.whisper import transcribe_audio

# --- image OCR / vision ---
from retie_agent.services.vision import ocr_image, vision_extract_insights


# Helper: run blocking code in executor **while preserving contextvars**
# so OTel/Langfuse spans keep parentage
async def _to_thread_ctx(func, *args, **kwargs):
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ctx.run(func, *args, **kwargs))


# Entrega con feedback: el "typing" de Telegram expira a los ~5s, así que en
# consultas largas (NotebookLM puede tardar >30s) el chat parecía muerto.
# Mantiene el typing vivo y, pasado un umbral, avisa que se está consultando.
_PROGRESS_NOTICE = (
    "🔎 Estoy consultando la base normativa y NotebookLM para darte una "
    "respuesta completa; puede tardar un poco más…"
)
_PROGRESS_AFTER_S = 10.0
_TYPING_REFRESH_S = 4.0


async def _run_graph_with_feedback(message: Message, *args, **kwargs):
    task = asyncio.create_task(_to_thread_ctx(run_graph, *args, **kwargs))
    elapsed = 0.0
    notified = False
    while True:
        done, _ = await asyncio.wait({task}, timeout=_TYPING_REFRESH_S)
        if done:
            return task.result()
        elapsed += _TYPING_REFRESH_S
        try:
            await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        except Exception:
            pass
        if not notified and elapsed >= _PROGRESS_AFTER_S:
            notified = True
            try:
                await message.answer(_PROGRESS_NOTICE)
            except Exception:
                pass


# Agente único: cualquier key legacy (auto/plumber/pymupdf/vigente/historica)
# va al comportamiento por defecto — la colección única "normativas".
def _safe_agent_key(k: Optional[str]) -> Optional[str]:
    if not k:
        return None
    if k.lower() in {"auto", "plumber", "pymupdf", "vigente", "historica"}:
        return None
    return k


# ----------------------- commands & admin -----------------------
_HELP_TEXT = (
    "🤖 <b>Bot RETIE — comandos disponibles</b>\n\n"
    "/start — Inicia la conversación\n"
    "/help — Muestra esta ayuda\n"
    "/clear — Borra el historial de esta conversación\n"
    "/admin &lt;contraseña&gt; — Acceso administrador\n"
    "/docs [k] — Documentos más relevantes de la última consulta (admin)\n"
    "/logout — Cierra la sesión de administrador\n\n"
    "Respondo sobre el RETIE y la NTC 2050 (Código Eléctrico Colombiano).\n"
    "También puedes enviar <b>notas de voz</b> e <b>imágenes</b> de documentos.\n"
    "💡 Pide una «tabla» para recibir los datos en formato tabular."
)


@router.message(CommandStart())
async def on_start(message: Message):
    CHAT_AGENT[message.chat.id] = DEFAULT_AGENT
    await message.answer("¡Hola! Soy tu bot RETIE, ¿en qué puedo ayudarte?\nEscribe /help para ver lo que puedo hacer.")


@router.message(Command("help"))
async def on_help(message: Message):
    await message.answer(_HELP_TEXT)


@router.message(Command("clear"))
async def on_clear(message: Message):
    """Borra el historial de la conversación (el mismo session_id que usa run_graph)."""
    deleted = await asyncio.to_thread(clear_history, f"telegram-chat-{message.chat.id}")
    if deleted:
        await message.answer(f"🧹 Historial borrado ({deleted} mensajes). Empezamos de cero.")
    else:
        await message.answer("No había historial que borrar en esta conversación.")


@router.message(Command("agent"))
async def on_agent(message: Message):
    parts = (message.text or "").strip().split()
    if len(parts) < 2:
        return await message.answer("Formato: /agent <key>")
    key = parts[1].lower()
    if key == "auto":
        CHAT_AGENT[message.chat.id] = "auto"
        return await message.answer("✅ Agente: auto (busca en todas las colecciones)")
    if _AGENTS and key not in _AGENTS:
        opciones = ", ".join(["auto", *_AGENTS.keys()])
        return await message.answer(f"Agente inválido. Opciones: {opciones}")
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
    """Muestra documentos top-k de la última consulta (solo admins)."""
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
        from retie_agent.retriever.retrieve import search
        from retie_agent.agent.retie_agent import _resolve_collection, _dedupe_hits

        coll = _resolve_collection(agent_key=_safe_agent_key(key), explicit=None)
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


async def _download_image_to_tmp(message: Message) -> Optional[Path]:
    """Download the highest-res photo or an image document to a temp file."""
    # Photo payload
    if message.photo:
        photo = message.photo[-1]
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        Path(tmp_path).unlink(missing_ok=True)
        dest = Path(tmp_path)
        await message.bot.download(photo, destination=dest)
        return dest
    # Image as document
    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        suffix = Path(message.document.file_name or "image.jpg").suffix or ".jpg"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        Path(tmp_path).unlink(missing_ok=True)
        dest = Path(tmp_path)
        await message.bot.download(message.document, destination=dest)
        return dest
    return None


def _compose_question_from_image(caption: str, ocr_txt: str, vision_txt: str) -> str:
    """Ensures the OCR/vision content is included in the prompt to the agent."""
    parts: List[str] = []
    if caption:
        parts.append(f"Usuario dijo sobre la imagen: {caption.strip()}")
    if ocr_txt:
        parts.append(f"Texto detectado en la imagen:\n{ocr_txt.strip()}")
    elif vision_txt:
        parts.append(f"Contenido interpretado de la imagen:\n{vision_txt.strip()}")
    if not parts:
        parts.append("Interpreta la imagen y responde según el RETIE.")

    parts.append("Con base en lo anterior, responde la consulta del usuario de forma breve y precisa.")
    return "\n\n".join(parts)


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
    except Exception:
        try:
            local_file.unlink(missing_ok=True)
            if wav_file and wav_file != local_file:
                wav_file.unlink(missing_ok=True)
        finally:
            return await message.answer("⚠️ audio inaudible")

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

    # Ejecuta LangGraph (con su propia traza 'LangGraph')
    raw_resp = await _run_graph_with_feedback(
        message,
        transcript,
        user_id=str(message.from_user.id) if message.from_user else "anon",
        session=f"telegram-chat-{message.chat.id}",
        agent_key=_safe_agent_key(agent_key),
        metadata={"via": "voice"},
    )

    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    await _send_response(message, raw_resp, is_admin)


# ----------------------- IMAGES (photo + image document) -----------------------
@router.message(F.photo)
async def on_photo(message: Message):
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    img_path = await _download_image_to_tmp(message)
    if not img_path:
        return await message.answer("No pude descargar la imagen.")

    ocr_lang = os.getenv("OCR_LANG", "eng")  # set OCR_LANG=spa if you installed Spanish data
    ocr_txt = ""
    try:
        ocr_txt = ocr_image(img_path, lang=ocr_lang)
    except Exception:
        ocr_txt = ""

    vision_txt = ""
    if not ocr_txt or len(ocr_txt) < 12:
        try:
            vision_txt = vision_extract_insights(
                img_path,
                user_prompt=message.caption or "",
                model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
            )
        except Exception:
            vision_txt = ""

    question = _compose_question_from_image(message.caption or "", ocr_txt, vision_txt)
    LAST_QUERY[message.chat.id] = question

    raw_resp = await _run_graph_with_feedback(
        message,
        question,
        user_id=str(message.from_user.id) if message.from_user else "anon",
        session=f"telegram-chat-{message.chat.id}",
        agent_key=_safe_agent_key(agent_key),
        metadata={"via": "image", "ocr_len": len(ocr_txt), "vision_len": len(vision_txt)},
    )

    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    await _send_response(message, raw_resp, is_admin)


@router.message(F.document)
async def on_image_document(message: Message):
    # Only handle if it's an image/*
    if not (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")):
        return  # ignore other documents
    return await on_photo(message)  # reuse same logic


# ----------------------- TEXT -----------------------
@router.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    LAST_QUERY[message.chat.id] = q

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    # Step 1️⃣: Run the LangGraph pipeline safely
    result = None
    try:
        result = await _run_graph_with_feedback(
            message,
            q,
            user_id=str(message.from_user.id) if message.from_user else "anon",
            session=f"telegram-chat-{message.chat.id}",
            agent_key=_safe_agent_key(agent_key),
            metadata={"via": "text", "channel": "telegram"},
        )
    except Exception as e:
        # Graceful fallback if graph execution fails
        print(f"[⚠️ run_graph error] {e}")
        result = {"formatted_response": f"⚠️ Ocurrió un error interno: {e}"}

    # Step 2️⃣: Entrega según rol; foto si el agente generó una imagen de tabla.
    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    await _send_response(message, result, is_admin)

