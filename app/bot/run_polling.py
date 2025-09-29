# app/bot/run_polling.py
import asyncio
import logging
import os
import shutil
from minio import Minio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
load_dotenv()

from app.bot.router import router

# ------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGS
# ------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, force=True)

# ------------------------------------------------------------------------
# CONFIGURACIÓN MINIO / BUCKET
# ------------------------------------------------------------------------
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "data")
MINIO_ENDPOINT = os.getenv("MINIO_PRIVATE_ENDPOINT", "bucket.railway.internal:9000")
ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
LOCAL_CHROMA_DIR = "./data/chroma_db"

# ------------------------------------------------------------------------
# DESCARGA DE EMBEDDINGS DESDE EL BUCKET (SOFT FAIL)
# ------------------------------------------------------------------------
try:
    if os.path.exists(LOCAL_CHROMA_DIR):
        shutil.rmtree(LOCAL_CHROMA_DIR)
    os.makedirs(LOCAL_CHROMA_DIR, exist_ok=True)

    client = Minio(
        MINIO_ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )

    logging.info("🔄 Descargando embeddings desde el bucket...")

    if not client.bucket_exists(BUCKET_NAME):
        logging.warning(f"⚠️ El bucket '{BUCKET_NAME}' no existe en MinIO")
    else:
        count = 0
        for obj in client.list_objects(BUCKET_NAME, recursive=True):
            dest_path = os.path.join(LOCAL_CHROMA_DIR, obj.object_name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            client.fget_object(BUCKET_NAME, obj.object_name, dest_path)
            logging.info(f"✅ Archivo descargado: {obj.object_name}")
            count += 1

        if count == 0:
            logging.warning(f"⚠️ El bucket '{BUCKET_NAME}' está vacío.")
        else:
            logging.info(f"✅ Descarga completa: {count} archivos en {LOCAL_CHROMA_DIR}")

except Exception as e:
    logging.warning(f"⚠️ No se pudieron descargar embeddings: {e}")
    logging.warning("➡️ El bot seguirá funcionando con datos locales si existen...")

# ------------------------------------------------------------------------
# CONFIGURACIÓN DEL BOT
# ------------------------------------------------------------------------
dp = Dispatcher()
dp.include_router(router)

async def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Falta TELEGRAM_TOKEN. El bot no podrá conectarse a Telegram.")
        # En vez de detener el contenedor, lo dejamos en un loop infinito suave
        while True:
            await asyncio.sleep(60)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))

    logging.info("🤖 Iniciando bot RETIE...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
