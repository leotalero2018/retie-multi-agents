# app/bot/run_polling.py
import asyncio
import logging
import os
from minio import Minio  # Asegúrate de tener 'minio' instalado: pip install minio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
load_dotenv()

from app.bot.router import router

# ------------------------------------------------------------------------
# CONFIGURACIÓN DEL BUCKET
# ------------------------------------------------------------------------
# Variables de entorno (definidas en Railway)
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "Data")  # Tu bucket
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")         # Host del bucket
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# Ruta local donde Chroma buscará los embeddings
LOCAL_CHROMA_DIR = "./data/chroma_db"
os.makedirs(LOCAL_CHROMA_DIR, exist_ok=True)

# ------------------------------------------------------------------------
# DESCARGA DE EMBEDDINGS DESDE EL BUCKET
# ------------------------------------------------------------------------
# Conexión al bucket
client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=True  
)

# Descarga todos los objetos del bucket a la carpeta local
for obj in client.list_objects(BUCKET_NAME, recursive=True):
    dest_path = os.path.join(LOCAL_CHROMA_DIR, obj.object_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    client.fget_object(BUCKET_NAME, obj.object_name, dest_path)

logging.info(f"✅ Embeddings descargados en {LOCAL_CHROMA_DIR}")

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
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
