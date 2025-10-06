# app/bot/run_polling.py
# Run with: python -m app.bot.run_polling
from __future__ import annotations

import asyncio
import logging
import os
import glob
from pathlib import Path

from dotenv import load_dotenv

# Load local env (dev). pydantic-settings also loads .env; harmless double-load.
load_dotenv()

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from app.config import settings
from app.bootstrap_sync import sync_chroma_from_minio
from app.bot.router import router

logging.basicConfig(level=logging.INFO, force=True)
dp = Dispatcher()
dp.include_router(router)


def _clear_if_placeholder(path: str) -> None:
    """If dir exists but lacks Chroma markers, wipe it so a sync can refill cleanly."""
    p = Path(path)
    sqlite = p / "chroma.sqlite3"
    has_shard = any(p.glob("*.db"))
    if p.exists() and p.is_dir() and (not sqlite.exists()) and (not has_shard):
        for child in p.iterdir():
            try:
                if child.is_file():
                    child.unlink()
                else:
                    import shutil
                    shutil.rmtree(child)
            except Exception as e:
                logging.warning("[BOOT] Could not remove %s: %s", child, e)
        logging.info("[BOOT] Cleared placeholder contents in %s", path)


async def main() -> None:
    # 1) Sync Chroma from MinIO into a guaranteed-writable runtime dir
    #    and point everything (env + settings) to that dir.
    try:
        runtime_dir = sync_chroma_from_minio()
    except Exception as e:
        logging.warning("[SYNC] MinIO sync raised: %s. Continuing with local paths.", e)
        runtime_dir = settings.CHROMA_DB_DIR  # fallback to whatever is configured

    if not runtime_dir:
        runtime_dir = settings.CHROMA_DB_DIR

    os.makedirs(runtime_dir, exist_ok=True)
    _clear_if_placeholder(runtime_dir)

    # Point Chroma paths to the writable runtime dir
    os.environ["CHROMA_DB_DIR"] = runtime_dir
    os.environ["CHROMA_PERSIST_DIR"] = runtime_dir
    settings.CHROMA_DB_DIR = runtime_dir  # ensure all imports use the same path

    logging.info("[BOOT] Using COLLECTION_NAME=%s  CHROMA_DIR=%s",
                 settings.COLLECTION_NAME, settings.CHROMA_DB_DIR)

    # 2) Quick listing (helps diagnose prefix/path issues)
    try:
        files = glob.glob(os.path.join(settings.CHROMA_DB_DIR, "*"))
        logging.info("[LS] %s -> %s", settings.CHROMA_DB_DIR, files[:30])
    except Exception as e:
        logging.error("[LS] ERROR listing files: %s", e)

    # 3) Deep check: list all collections and counts from on-disk DB
    try:
        import chromadb
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

    # 4) Start Telegram bot (supports TELEGRAM_BOT_TOKEN or TELEGRAM_TOKEN)
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Missing TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN). Bot cannot start.")
        while True:
            await asyncio.sleep(60)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    logging.info("🤖 Iniciando bot RETIE..")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
