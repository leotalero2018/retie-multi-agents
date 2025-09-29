# app/retriever/chroma_client.py
from typing import Any
import chromadb
from chromadb import PersistentClient
from app.config import settings

# Create a single persistent Chroma client pointing to the same directory used at runtime
_client: PersistentClient = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

# Local cache of collections (per-process)
_collections: dict[str, Any] = {}


def get_collection(name: str | None = None):
    """
    Returns or creates a collection by name and caches it in this process.
    IMPORTANT:
      - Do NOT attach any embedding_function here to avoid EF conflicts
        with persisted/default collection configuration.
      - Distance space is set via metadata only.
    """
    col_name = name or settings.COLLECTION_NAME

    if col_name not in _collections:
        # Ensure collection exists WITHOUT embedding_function to prevent conflicts.
        # Use cosine space so distances are comparable to cosine similarity thresholds.
        try:
            col = _client.get_collection(col_name)
        except Exception:
            col = _client.create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )
        _collections[col_name] = col

    return _collections[col_name]


def drop_collection(name: str):
    """Drops the collection entirely and clears local cache."""
    try:
        _client.delete_collection(name)
        print(f"✅ Collection {name} deleted.")
    except Exception:
        print(f"⚠ Collection {name} did not exist; continuing.")
    _collections.pop(name, None)
