# app/bot/run_polling.py
# Run with: python -m app.bot.run_polling
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load local env first (dev); pydantic-settings will also load .env
load_dotenv()

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from app.config import settings
from app.bootstrap_sync import sync_chroma_from_minio
from app.bot.router import router

import os, logging
from app.config import settings

logging.info("[BOOT] Using COLLECTION_NAME=%s  CHROMA_DIR=%s",
             settings.COLLECTION_NAME, settings.CHROMA_DB_DIR)


# ------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, force=True)

# Dispatcher + routes
dp = Dispatcher()
dp.include_router(router)


def _clear_if_placeholder(path: str) -> None:
    """
    If the directory exists but does NOT contain a real Chroma DB marker,
    wipe its contents so the sync can refill it cleanly.
    """
    p = Path(path)
    sqlite = p / "chroma.sqlite3"
    # Additionally, check for shard folders like "<uuid>.db"
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
    # Ensure persist dir exists (POSIX path on Railway)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    # If dir exists but has no real DB files, clear it to allow a clean sync
    _clear_if_placeholder(settings.CHROMA_PERSIST_DIR)

    # (3) Try to sync from MinIO; if it fails, continue with local DB
    try:
        logging.info("[SYNC] Intentando sincronizar Chroma DB desde MinIO...")
        sync_chroma_from_minio()
    except Exception as e:
        logging.warning(
            "[SYNC] Falló la sincronización desde MinIO (%s). Continuando con la base de datos local.",
            e
        )

    # List files present (helps diagnose prefix/path issues)
    try:
        import glob
        files = glob.glob(os.path.join(settings.CHROMA_PERSIST_DIR, "*"))
        logging.info("[LS] %s -> %s", settings.CHROMA_PERSIST_DIR, files[:20])
    except Exception as e:
        logging.error("[LS] ERROR listing files: %s", e)

    # Sanity check: print collection count so we know DB is actually there
    try:
        from app.retriever.chroma_client import get_collection
        col = get_collection(settings.COLLECTION_NAME)
        logging.info(
            "[CHK] Collection='%s' count=%s  dir=%s",
            settings.COLLECTION_NAME,
            col.count(),
            settings.CHROMA_PERSIST_DIR,
        )
    except Exception as e:
        logging.error("[CHK] ERROR checking collection '%s': %s", settings.COLLECTION_NAME, e)

    # Deep diag: list all collections and counts from the on-disk DB
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

    # Token: support new TELEGRAM_BOT_TOKEN and legacy TELEGRAM_TOKEN
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Missing TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN). Bot cannot start.")
        # Keep process alive for health checks without CPU burn
        while True:
            await asyncio.sleep(60)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    logging.info("🤖 Iniciando bot RETIE..")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

