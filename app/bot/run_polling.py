# app/bot/run_polling.py
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

from app.config import settings
from app.agent.registry import AGENTS
from app.agent.graph import run_graph
from app.retriever.chroma_client import get_collection

bot = Bot(token=settings.TELEGRAM_TOKEN)
dp = Dispatcher()

CHAT_AGENT: dict[int, str] = {}
DEFAULT_AGENT = "plumber"


@dp.message(CommandStart())
async def on_start(message: Message):
    CHAT_AGENT[message.chat.id] = DEFAULT_AGENT
    keys = ", ".join(AGENTS.keys()) if AGENTS else "plumber,pymupdf,hybrid"
    await message.answer(
        "¡Hola! Soy tu bot RETIE.\n"
        f"Agentes disponibles: {keys}\n"
        f"Usa /agent <key> para cambiar. Actual: {CHAT_AGENT[message.chat.id]}"
    )


@dp.message(Command("agent"))
async def on_agent(message: Message):
    parts = (message.text or "").strip().split()
    if len(parts) < 2:
        return await message.answer(f"Formato: /agent <key>\nOpciones: {', '.join(AGENTS.keys())}")
    key = parts[1].lower()
    if key not in AGENTS:
        return await message.answer(f"Agente inválido. Usa uno de: {', '.join(AGENTS.keys())}")
    CHAT_AGENT[message.chat.id] = key
    cfg = AGENTS[key]
    await message.answer(f"✅ Agente: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})")


@dp.message(Command("who"))
async def on_who(message: Message):
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    cfg = AGENTS[key]
    await message.answer(f"Agente actual: {key} (modelo={cfg.resolved_chat_model()}, colección={cfg.collection})")


@dp.message(Command("debug"))
async def on_debug(message: Message):
    key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    cfg = AGENTS[key]
    col = get_collection(cfg.collection)
    try:
        count = col.count()
        peek = col.peek()
        peek_keys = list(peek.keys()) if isinstance(peek, dict) else type(peek)
    except Exception as e:
        count, peek_keys = f"error: {e}", "n/a"
    await message.answer(
        f"CHROMA_DB_DIR={settings.CHROMA_DB_DIR}\n"
        f"Agente={key}\nColección={cfg.collection}\n"
        f"Count={count}\nPeek keys={peek_keys}"
    )


@dp.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)
    user_id = str(message.from_user.id) if message.from_user else None
    session_id = f"tg:{message.chat.id}"

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    loop = asyncio.get_running_loop()
    # run_graph(question, user_id, session_id, agent_key)
    resp = await loop.run_in_executor(None, run_graph, q, user_id, session_id, agent_key)
    await message.answer(resp)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
