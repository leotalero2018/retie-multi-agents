# app/bot/run_polling.py
from __future__ import annotations
import asyncio
import logging
import os
import glob
from pathlib import Path
from dotenv import load_dotenv

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
    # 1️⃣ Sync from MinIO → local runtime dir
    try:
        runtime_dir = sync_chroma_from_minio() or settings.CHROMA_DB_DIR
    except Exception as e:
        logging.warning("[SYNC] MinIO sync failed: %s. Using local directory.", e)
        runtime_dir = settings.CHROMA_DB_DIR

    os.environ["CHROMA_DB_DIR"] = runtime_dir
    os.environ["CHROMA_PERSIST_DIR"] = runtime_dir
    settings.CHROMA_DB_DIR = runtime_dir

    from app.retriever import chroma_client as cc
    cc.set_persist_dir(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    _clear_if_placeholder(runtime_dir)

    logging.info("[BOOT] Using COLLECTION_NAME=%s | CHROMA_DIR=%s",
                 settings.COLLECTION_NAME, settings.CHROMA_DB_DIR)

    # 2️⃣ Quick check
    try:
        files = glob.glob(os.path.join(settings.CHROMA_DB_DIR, "*"))
        logging.info("[LS] Found %d files under %s", len(files), settings.CHROMA_DB_DIR)
    except Exception as e:
        logging.error("[LS] ERROR listing files: %s", e)

    # 3️⃣ Optional: list Chroma collections
    try:
        import chromadb
        cli = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        cols = cli.list_collections()
        for c in cols:
            try:
                cnt = cli.get_collection(c.name).count()
            except Exception as e:
                cnt = f"error: {e}"
            logging.info("   - %s: %s items", c.name, cnt)
    except Exception as e:
        logging.warning("[CHK2] Skipped Chroma check: %s", e)

    # 4️⃣ Start Telegram bot
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Missing TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN). Cannot start bot.")
        while True:
            await asyncio.sleep(60)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    logging.info("🤖 Starting RETIE bot polling loop...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
