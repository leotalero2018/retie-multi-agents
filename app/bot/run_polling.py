#Despliegue del agente  con el uso de .env (token de Telegram y motor de Openai)

# app/bot/run_polling.py (añade imports)
import tempfile
import asyncio
from pathlib import Path
from aiogram.enums import ChatAction
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from app.agent.graph import run_graph
from pydub import AudioSegment

from app.config import settings
from app.agent.orchestrator import answer_question
from app.agent.registry import AGENTS
from app.services.whisper import transcribe_audio
from app.utils.storage import put_file, build_object_name

bot = Bot(token=settings.TELEGRAM_TOKEN)
dp = Dispatcher()

CHAT_AGENT: dict[int, str] = {}    # chat_id -> agent_key
DEFAULT_AGENT = "plumber"          # debe existir en AGENTS


@dp.message(CommandStart())
async def on_start(message: Message):
    CHAT_AGENT[message.chat.id] = DEFAULT_AGENT
    keys = ", ".join(AGENTS.keys())
    await message.answer(
        "Hola! Soy tu bot RETIE.\n"
        f"En que te puedo colaborar?"
        f"Agentes disponibles: {keys}\n"
        f"Está usando actualmente /agent <key>. {CHAT_AGENT[message.chat.id]}"
    )


@dp.message(Command("agent"))
async def on_agent(message: Message):
    parts = (message.text or "").strip().split()
    if len(parts) < 2:
        await message.answer(f"Formato: /agent <key>\nOpciones: {', '.join(AGENTS.keys())}")
        return
    key = parts[1].lower()
    if key not in AGENTS:
        await message.answer(f"Agente inválido. Usa uno de: {', '.join(AGENTS.keys())}")
        return
    CHAT_AGENT[message.chat.id] = key
    cfg = AGENTS[key]
    await message.answer(f"✅ Agente: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})")


@dp.message(Command("who"))
async def on_who(message: Message):
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    cfg = AGENTS[key]
    await message.answer(f"Agente actual: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})")


@dp.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    user_id = str(message.from_user.id)
    session = f"telegram-chat-{message.chat.id}"

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, run_graph, q, user_id, session)

    await message.answer(resp)


@dp.message(F.voice)
async def on_voice(message: Message):
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    # 1) Descargar audio OGG/OPUS a tmp
    file = await bot.get_file(message.voice.file_id)
    with tempfile.TemporaryDirectory() as td:
        ogg_path = Path(td) / "audio.ogg"
        wav_path = Path(td) / "audio.wav"
        await bot.download(file, destination=ogg_path)

        # 2) Convertir a WAV (16kHz mono de ser posible)
        audio = AudioSegment.from_file(ogg_path)  # auto-detecta OGG/OPUS si ffmpeg está en PATH
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

        # 3) Transcribir
        text = transcribe_audio(wav_path, language="es")

    # 4) (Opcional) Pasa la transcripción al agente activo
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, answer_question, text, None, None, key)

    await message.answer(f"🗣️ *Transcripción:*\n{text}", parse_mode="Markdown")
    await message.answer(resp)


@dp.message(F.photo)
async def on_photo(message: Message):
    await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)

    photo = message.photo[-1]  # mayor resolución
    tg_file = await bot.get_file(photo.file_id)

    with tempfile.TemporaryDirectory() as td:
        local_path = Path(td) / "image.jpg"
        await bot.download(tg_file, destination=local_path)

        # Subir a MinIO
        object_name = build_object_name(message.chat.id, prefix="images", suffix="jpg")
        url = put_file(settings.MINIO_BUCKET_IMAGES, local_path, object_name=object_name)

    await message.answer(f"🖼️ Imagen guardada.\nURL (temporal): {url}")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
