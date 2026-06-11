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

    # --- NotebookLM MCP ---
    NOTEBOOKLM_ENABLED: str = "false"          # "true" | "1" | "yes" to enable
    NOTEBOOKLM_URL: str = "http://localhost:3000"
    NOTEBOOKLM_NOTEBOOK_ID: Optional[str] = None  # share link or notebook ID
    NOTEBOOKLM_TIMEOUT: float = 120.0
    # Per-browser-operation timeout (ms). Lower than the server's 30s default so a
    # stuck click (overlay intercepting the textarea) fails fast instead of hanging.
    NOTEBOOKLM_PAGE_TIMEOUT_MS: int = 15000
    # Hybrid RAG settings
    NOTEBOOKLM_CACHE_TTL: int = 3600          # seconds to cache NLM responses in memory
    NOTEBOOKLM_PARALLEL_TIMEOUT: float = 45.0  # max seconds to wait for NLM in parallel mode
    CHROMA_HIGH_CONFIDENCE_THR: float = 0.25   # skip NLM when Chroma best score < this
    # Session persistence (Playwright user-data-dir backed up in MinIO)
    NOTEBOOKLM_SESSION_DIR: str = "/tmp/nlm-session"  # where Playwright stores the browser profile
    NOTEBOOKLM_SESSION_BUCKET: str = ""               # MinIO bucket (defaults to S3_BUCKET_NAME)
    NOTEBOOKLM_STARTUP_TIMEOUT: int = 60              # seconds to wait for MCP server to become ready

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
    repo_root = Path(__file__).resolve().parents[1]
    persist_abs = str((repo_root / persist).resolve())
else:
    persist_abs = persist

settings.CHROMA_PERSIST_DIR = persist_abs
# Maintain legacy attribute for any old imports
settings.CHROMA_DB_DIR = persist_abs

# Normalize Telegram token (prefer new name)
if not settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_TOKEN:
    settings.TELEGRAM_BOT_TOKEN = settings.TELEGRAM_TOKEN

# --- Langfuse: reconciliar región (host vs base_url) ---
# El SDK de Langfuse da PRECEDENCIA a la env var LANGFUSE_BASE_URL por encima
# del kwarg `host`. Si LANGFUSE_BASE_URL apunta a una región distinta de
# LANGFUSE_HOST (p.ej. EU vs US), el SDK envía las trazas a la región
# equivocada y el servidor responde 401 ("Invalid credentials / correct host").
# Forzamos que ambas env vars coincidan con LANGFUSE_HOST para que TODA
# instancia de Langfuse del proceso (cliente propio + CallbackHandler) use la
# misma región, sin depender de la config externa (Railway).
if settings.LANGFUSE_HOST:
    _lf_host = settings.LANGFUSE_HOST.strip().strip('"').strip("'").strip()
    _lf_stale = os.environ.get("LANGFUSE_BASE_URL")
    if _lf_stale and _lf_stale.strip().rstrip("/") != _lf_host.rstrip("/"):
        import logging
        logging.getLogger(__name__).warning(
            "[Langfuse] LANGFUSE_BASE_URL=%r contradecía LANGFUSE_HOST=%r — "
            "normalizado a LANGFUSE_HOST para evitar 401 por región incorrecta.",
            _lf_stale, _lf_host,
        )
    os.environ["LANGFUSE_HOST"] = _lf_host
    os.environ["LANGFUSE_BASE_URL"] = _lf_host
    settings.LANGFUSE_HOST = _lf_host

# --- Configuración del asistente de enriquecimiento ---
ENRICHMENT_ASSISTANT_ID = os.getenv(
    "ENRICHMENT_ASSISTANT_ID", "asst_qthy1ZfTpr2ps0mruX30zVlc"
)
ENRICHMENT_VECTOR_STORE_ID = os.getenv(
    "ENRICHMENT_VECTOR_STORE_ID", "vs_69050fe6e43c8191be28bac47c3f565f"
)
