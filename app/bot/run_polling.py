# app/bot/run_polling.py
import asyncio
import logging
import os
from minio import Minio  # pip install minio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
load_dotenv()

from app.bot.router import router

# ------------------------------------------------------------------------
# CONFIGURACIÓN MINIO / BUCKET
# ------------------------------------------------------------------------
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "Data")

# 🔹 Debe ser solo host, sin https:// ni puerto
raw_endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT", "bucket-production-b0dd.up.railway.app")
MINIO_ENDPOINT = raw_endpoint.replace("https://", "").replace("http://", "").split(":")[0]

ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")

# Carpeta local donde Chroma buscará los embeddings
LOCAL_CHROMA_DIR = "./data/chroma_db"
os.makedirs(LOCAL_CHROMA_DIR, exist_ok=True)

# ------------------------------------------------------------------------
# DESCARGA DE EMBEDDINGS DESDE EL BUCKET
# ------------------------------------------------------------------------
try:
    client = Minio(
        MINIO_ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=True  # HTTPS
    )

    logging.info("🔄 Descargando embeddings desde el bucket...")

    for obj in client.list_objects(BUCKET_NAME, recursive=True):
        dest_path = os.path.join(LOCAL_CHROMA_DIR, obj.object_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        client.fget_object(BUCKET_NAME, obj.object_name, dest_path)
        logging.info(f"✅ Archivo descargado: {obj.object_name}")

    logging.info(f"✅ Descarga completa en {LOCAL_CHROMA_DIR}")

except Exception as e:
    logging.warning(f"⚠️ No se pudieron descargar embeddings desde el bucket: {e}")
    logging.warning("Continuando ejecución del bot con datos locales si existen...")

# ------------------------------------------------------------------------
# CONFIGURACIÓN DEL BOT
# ------------------------------------------------------------------------
dp = Dispatcher()
dp.include_router(router)

async def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_TOKEN")

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))

    logging.basicConfig(level=logging.INFO)
    logging.info("🤖 Iniciando bot RETIE...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
