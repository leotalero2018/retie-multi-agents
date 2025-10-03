# app/bot/run_polling.py
# Run with: python -m app.bot.run_polling
from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

# Load local env first (dev); pydantic-settings also loads .env but this is harmless
load_dotenv()

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from app.config import settings
from app.bootstrap_sync import sync_chroma_from_minio
from app.bot.router import router

# ------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, force=True)

# Dispatcher + routes
dp = Dispatcher()
dp.include_router(router)

async def main() -> None:
    # Ensure persist dir exists (POSIX path on Railway)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    # Sync Chroma DB from MinIO (only downloads if DB isn't ready or MINIO_FORCE_SYNC=true)
    sync_chroma_from_minio()

    # Sanity check: print collection count so we know DB is actually there
    try:
        from app.retriever.chroma_client import get_collection
        col = get_collection(settings.COLLECTION_NAME)
        logging.info(
            f"[CHK] Collection='{settings.COLLECTION_NAME}' "
            f"count={col.count()}  dir={settings.CHROMA_PERSIST_DIR}"
        )
    except Exception as e:
        logging.error("[CHK] ERROR checking collection: %s", e)

    # Deep diag: listar todas las colecciones y sus counts en la BD en disco
    try:
        import chromadb
        from app.config import settings

        cli = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        cols = cli.list_collections()
        if not cols:
            logging.error("[CHK2] No collections found in DB at %s", settings.CHROMA_DB_DIR)
        else:
            logging.info("[CHK2] Collections present in DB:")
            for c in cols:
                try:
                    cnt = cli.get_collection(c.name).count()
                except Exception as e:
                    cnt = f"error: {e}"
                logging.info("   - name=%s  count=%s", c.name, cnt)
    except Exception as e:
        logging.error("[CHK2] ERROR listing collections: %s", e)
        

    # Token: support new TELEGRAM_BOT_TOKEN and legacy TELEGRAM_TOKEN
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Missing TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN). Bot cannot start.")
        # Keep process alive for health checks without CPU burn
        while True:
            await asyncio.sleep(60)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    logging.info("🤖 Iniciando bot RETIE...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
