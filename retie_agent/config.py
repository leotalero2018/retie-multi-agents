# app/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


def as_bool(value: object) -> bool:
    """Interpreta flags tipo string ("true"/"1"/"yes") de forma consistente."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ── Saneo del entorno ANTES de cargar Settings ────────────────────────────────
# Caso real en Railway: `NOTEBOOKLM_ENABLED ="true"` (espacio antes del `=`)
# crea una env var llamada "NOTEBOOKLM_ENABLED " que pydantic nunca encuentra,
# dejando el flag en su default. Se duplica bajo el nombre limpio.
for _k in list(os.environ):
    _ks = _k.strip()
    if _ks and _ks != _k and _ks not in os.environ:
        os.environ[_ks] = os.environ[_k]


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
    COLLECTION_NAME: str = "retie_docs"            # legacy single-collection (compat)
    # Colección(es) que consulta el agente (admite varias separadas por coma).
    # El pipeline (pipeline_indexacion/indexar_normativas.py) indexa TODO en la
    # colección única "normativas"; si no existe, el retriever cae a las
    # colecciones disponibles en la DB.
    COLLECTION_NAMES: str = "normativas"
    TOP_K: int = 4
    CHUNK_TOKENS: int = 800
    CHUNK_OVERLAP: int = 150
    RAG_DISTANCE_THRESHOLD: float = 0.45           # cosine distance filter
    # Candidatos extra que pide el dense retrieval antes de filtrar/fusionar.
    FETCH_K_MULTIPLIER: int = 3
    # Fusión léxica BM25 + dense vía Reciprocal Rank Fusion.
    BM25_FUSION_ENABLED: str = "true"
    RRF_K: int = 60                                # constante k del RRF

    # --- Conversación ---
    HISTORY_LIMIT: int = 10
    # Reescribe preguntas de seguimiento ("¿y eso aplica a...?") usando el
    # historial para que el retrieval reciba una pregunta autocontenida.
    QUERY_REWRITE_ENABLED: str = "true"

    # --- Enriquecimiento (OpenAI Assistants) ---
    ENRICHMENT_ENABLED: str = "true"               # "false" → respuesta ~2x más rápida
    ENRICHMENT_TIMEOUT: float = 30.0               # segundos máx. para el assistant

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
    # Modo "siempre ambos": la respuesta SIEMPRE combina Chroma + NotebookLM y
    # se espera a NLM hasta NOTEBOOKLM_HARD_TIMEOUT, sin importar la confianza
    # de Chroma. Con "false" se activa la espera adaptativa (timeouts de abajo).
    NOTEBOOKLM_ALWAYS_WAIT: str = "true"
    NOTEBOOKLM_HARD_TIMEOUT: float = 180.0     # tope absoluto de espera a NLM
    NOTEBOOKLM_PARALLEL_TIMEOUT: float = 25.0  # max seconds to wait for NLM in parallel mode
    CHROMA_HIGH_CONFIDENCE_THR: float = 0.25   # skip NLM when Chroma best score < this
    # Espera adaptativa: con evidencia "decente" de Chroma (mejor distancia <
    # CHROMA_DECENT_THR) solo se espera a NLM el soft timeout; la respuesta
    # tardía de NLM igual se cachea para la siguiente pregunta similar.
    CHROMA_DECENT_THR: float = 0.40
    NOTEBOOKLM_SOFT_TIMEOUT: float = 12.0
    # Las tablas dependen de NLM (Chroma rara vez tiene la tabla completa):
    # conservan un presupuesto de espera mayor.
    NOTEBOOKLM_TABLE_TIMEOUT: float = 45.0
    # Caché semántico: reutiliza la respuesta NLM de una pregunta previa cuya
    # similitud coseno de embeddings supere este umbral (0 = desactivado).
    NLM_SEMANTIC_CACHE_SIM: float = 0.93
    # NLM solo se consulta para preguntas con al menos estos caracteres.
    NOTEBOOKLM_MIN_QUERY_CHARS: int = 12
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

# ── Saneo de valores ──────────────────────────────────────────────────────────
# El editor raw de Railway conserva comillas y espacios como parte del VALOR
# (`URL="http://x:3000 "` llega con comillas y espacio final). Un espacio en la
# URL de NotebookLM o una comilla en una key rompen la integración en silencio.
def _clean_setting_str(v: str) -> str:
    s = v.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


for _name in type(settings).model_fields:
    _val = getattr(settings, _name, None)
    if isinstance(_val, str):
        _cleaned = _clean_setting_str(_val)
        if _cleaned != _val:
            setattr(settings, _name, _cleaned)

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
