# app/bot/run_polling.py
# Run with: python -m app.bot.run_polling
from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from app.config import settings
from app.bootstrap_sync import sync_chroma_from_minio
from app.bot.router import router

os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
sync_chroma_from_minio()

# Load env (local dev)
load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)

# Optional: auto-sync Chroma from S3 bucket at startup
def _maybe_sync_from_bucket():
    if os.getenv("AUTO_SYNC_FROM_BUCKET", "false").lower() not in ("1", "true", "yes"):
        return
    try:
        from app.services.storage_s3 import download_folder
        bucket = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_PREFIX", "chroma_db/")
        persist = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        if not bucket:
            logging.warning("AUTO_SYNC_FROM_BUCKET is enabled but S3_BUCKET_NAME is missing.")
            return
        logging.info(f"🔄 Syncing Chroma from s3://{bucket}/{prefix} -> {persist}")
        download_folder(bucket=bucket, prefix=prefix, local_dir=persist)
        logging.info("✅ Sync complete.")
    except Exception as e:
        logging.error(f"⚠️ Could not sync from bucket: {e}")

dp = Dispatcher()
dp.include_router(router)

async def main() -> None:
    # Support both TELEGRAM_BOT_TOKEN (new) and TELEGRAM_TOKEN (legacy)
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Missing TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN). Bot cannot start.")
        # Keep process alive for container health checks without hammering CPU
        while True:
            await asyncio.sleep(60)

    _maybe_sync_from_bucket()

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    logging.info("🤖 Iniciando bot RETIE...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
