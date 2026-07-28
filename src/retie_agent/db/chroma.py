# app/db/chroma.py
# Minimal, dependency-free Chroma client.
# - No LangChain imports
# - Uses the persisted embeddings already stored in Chroma
# - Exposes a small adapter with similarity_search(query, k)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Backward-compat for path envs
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "retie_docs")


@dataclass
class _Doc:
    """Lightweight result compatible with LangChain-style usage."""
    page_content: str
    metadata: Dict[str, Any]


class _VectorStoreAdapter:
    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def similarity_search(self, query: str, k: int = 5) -> List[_Doc]:
        """Return top-k documents as simple _Doc objects (page_content + metadata)."""
        res = self._collection.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []

        out: List[_Doc] = []
        for text, meta in zip(docs, metas):
            out.append(_Doc(page_content=text, metadata=meta or {}))
        return out

    # Optional accessors if you ever need raw chroma
    @property
    def collection(self):
        return self._collection

    @property
    def client(self):
        return self._client


def get_vector_store() -> _VectorStoreAdapter:
    """
    Initialize and return a minimal vector store adapter backed by Chroma.
    Uses persisted embeddings; does NOT compute embeddings here.
    """
    return _VectorStoreAdapter(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
