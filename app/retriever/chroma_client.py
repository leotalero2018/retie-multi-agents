# app/retriever/chroma_client.py
from __future__ import annotations

import os
from typing import Any, Dict

import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings

# Support both new and legacy envs
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
_DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME", "retie_docs")

# Single persistent client for the process
_client: PersistentClient = chromadb.PersistentClient(
    path=_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# Per-process collection cache
_collections: Dict[str, Any] = {}


def get_collection(name: str | None = None):
    """
    Get or create a Chroma collection by name; cached per process.
    IMPORTANT:
      - Do NOT attach embedding_function; we store embeddings explicitly.
      - Use cosine space so distances are interpretable vs. a cosine threshold.
    """
    col_name = name or _DEFAULT_COLLECTION

    if col_name not in _collections:
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
    """Drop the collection and clear local cache (safe if missing)."""
    try:
        _client.delete_collection(name)
        print(f"✅ Collection {name} deleted.")
    except Exception:
        print(f"⚠ Collection {name} not found; skipping delete.")
    _collections.pop(name, None)
