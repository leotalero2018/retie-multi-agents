from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Claves externas
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key")
    TELEGRAM_TOKEN: str | None = Field(default=None, description="Telegram bot token")

    # Proveedores (switch por .env)
    EMBEDDING_PROVIDER: str = "local"       # opciones: "local" | "openai"
    CHAT_PROVIDER: str = "extractive"       # opciones: "extractive" | "openai"

    # RAG / Chroma
    CHROMA_DB_DIR: str = "data/chroma_db"
    COLLECTION_NAME: str = "retie_docs"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"
    TOP_K: int = 4
    CHUNK_TOKENS: int = 800
    CHUNK_OVERLAP: int = 150
    MAX_TOKENS: int = 600

    WHISPER_PROVIDER: str = "openai"
    WHISPER_MODEL: str = "whisper-1"

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_IMAGES: str = "retie-images"
    MINIO_BUCKET_FILES: str = "retie-files"

    # pydantic-settings v2
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # por si aparecen variables extra en .env
    )

settings = Settings()
