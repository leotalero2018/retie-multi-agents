# app/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # --- Providers / keys ---
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_BASE_URL: Optional[str] = None  # optional (Azure/OpenRouter/etc.)
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, description="Telegram bot token")  # preferred
    TELEGRAM_TOKEN: Optional[str] = None  # legacy (kept for compatibility)

    # --- LLM / Embeddings ---
    EMBEDDING_PROVIDER: str = "openai"             # "openai" | "local"
    CHAT_PROVIDER: str = "openai"                  # "openai" | "extractive"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 600

    # --- RAG / Chroma ---
    CHROMA_PERSIST_DIR: str = "data/chroma_db"     # primary
    CHROMA_DB_DIR: Optional[str] = None            # legacy env; normalized to the same path
    COLLECTION_NAME: str = "retie_docs"
    TOP_K: int = 4
    CHUNK_TOKENS: int = 800
    CHUNK_OVERLAP: int = 150
    RAG_DISTANCE_THRESHOLD: float = 0.45           # cosine distance filter

    # --- Whisper ---
    WHISPER_PROVIDER: str = "openai"
    WHISPER_MODEL: str = "gpt-4o-transcribe"       # or "whisper-1"

    # --- Observability (Langfuse) ---
    # Accept typical string booleans; obs.py interprets them safely.
    LANGFUSE_ENABLED: str = "false"
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = None

    # --- S3-compatible storage (Railway/MinIO/Spaces/S3) ---
    S3_ENDPOINT_URL: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_REGION: str = "auto"

    # --- pydantic-settings v2 config ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# ---- Post-load normalization / compatibility ----
# Prefer CHROMA_PERSIST_DIR; keep CHROMA_DB_DIR as a mirror for legacy imports.
persist = settings.CHROMA_PERSIST_DIR or settings.CHROMA_DB_DIR or "data/chroma_db"

# Normalize to absolute path (repo root is two levels above this file)
if not os.path.isabs(persist):
    repo_root = Path(__file__).resolve().parents[2]
    persist_abs = str((repo_root / persist).resolve())
else:
    persist_abs = persist

settings.CHROMA_PERSIST_DIR = persist_abs
# Maintain legacy attribute for any old imports
settings.CHROMA_DB_DIR = persist_abs

# Normalize Telegram token (prefer new name)
if not settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_TOKEN:
    settings.TELEGRAM_BOT_TOKEN = settings.TELEGRAM_TOKEN
