# app/bot/run_polling.py
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

# Carga variables de entorno
from dotenv import load_dotenv
load_dotenv()

from app.bot.router import router

dp = Dispatcher()
dp.include_router(router)

async def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_TOKEN")

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))

    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
