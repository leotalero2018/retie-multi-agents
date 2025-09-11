# app/config.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # --- Proveedores / claves ---
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    TELEGRAM_TOKEN: Optional[str] = Field(default=None, description="Telegram bot token")

    EMBEDDING_PROVIDER: str = "local"        # "local" | "openai"
    CHAT_PROVIDER: str = "extractive"        # "extractive" | "openai"

    # --- RAG / Chroma ---
    CHROMA_DB_DIR: str = "data/chroma_db"
    COLLECTION_NAME: str = "retie_docs"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"
    TOP_K: int = 4
    CHUNK_TOKENS: int = 800
    CHUNK_OVERLAP: int = 150
    MAX_TOKENS: int = 600

    # --- Whisper ---
    WHISPER_PROVIDER: str = "openai"
    WHISPER_MODEL: str = "whisper-1"

    # --- MinIO ---
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_IMAGES: str = "retie-images"
    MINIO_BUCKET_FILES: str = "retie-files"

    # --- Config de pydantic-settings v2 ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignora variables desconocidas en .env
    )

settings = Settings()

# app/config.py (al final del archivo, después de settings = Settings())
from pathlib import Path
import os

# Normaliza CHROMA_DB_DIR a absoluta
_db_dir = settings.CHROMA_DB_DIR
if not os.path.isabs(_db_dir):
    # raíz = dos niveles arriba (raíz del repo)
    root = Path(__file__).resolve().parents[2]
    settings.CHROMA_DB_DIR = str((root / _db_dir).resolve())

