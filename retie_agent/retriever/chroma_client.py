# app/retriever/chroma_client.py
from __future__ import annotations
from typing import Any, Optional
import os
import chromadb
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from retie_agent.config import settings

# Lazily-initialized client bound to the *current* CHROMA_DB_DIR
_client: Optional[PersistentClient] = None
_client_path: Optional[str] = None

# If you really want to allow creating collections on the fly, set env ALLOW_CHROMA_CREATE=true
_ALLOW_CREATE = os.getenv("ALLOW_CHROMA_CREATE", "false").lower() in ("1", "true", "yes")


def _ensure_client() -> PersistentClient:
    """Create or re-create the PersistentClient if the path changed."""
    global _client, _client_path
    target_path = settings.CHROMA_DB_DIR
    if not target_path:
        # hard fallback
        target_path = "./data/chroma_db"
        settings.CHROMA_DB_DIR = target_path

    if (_client is None) or (_client_path != target_path):
        _client = chromadb.PersistentClient(path=target_path)
        _client_path = target_path
    return _client


def set_persist_dir(new_path: str) -> None:
    """
    Re-point the client to a new directory (e.g., after MinIO sync).
    Next call to _ensure_client() will open this path.
    """
    global _client, _client_path
    if new_path and new_path != settings.CHROMA_DB_DIR:
        settings.CHROMA_DB_DIR = new_path
    # force re-create on next use
    _client = None
    _client_path = None


def get_collection(name: str | None = None):
    """
    Return an *existing* collection. By default, DO NOT create collections here to avoid
    writing into a read-only/immutable DB. Opt-in creation with ALLOW_CHROMA_CREATE=true.
    """
    col_name = name or settings.COLLECTION_NAME
    cli = _ensure_client()

    try:
        return cli.get_collection(col_name)
    except NotFoundError:
        if _ALLOW_CREATE:
            # Only create if explicitly allowed
            return cli.create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )
        # Be explicit so callers know the DB is missing the expected collection
        raise RuntimeError(
            f"Chroma collection '{col_name}' not found at {settings.CHROMA_DB_DIR} "
            f"(set ALLOW_CHROMA_CREATE=true if you intend to create it here)."
        )


def drop_collection(name: str):
    cli = _ensure_client()
    try:
        cli.delete_collection(name)
        print(f"✅ Collection {name} deleted.")
    except Exception:
        print(f"⚠ Collection {name} did not exist; continuing.")
