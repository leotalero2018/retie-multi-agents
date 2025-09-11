# app/retriever/chroma_client.py
from typing import Any
from chromadb import PersistentClient
from app.config import settings

_client = PersistentClient(path=settings.CHROMA_DB_DIR)
_collections: dict[str, any] = {}


def get_collection(name: str | None = None):
    """
    Devuelve/crea una colección por nombre y la cachea en este proceso.
    Si name es None, usa settings.COLLECTION_NAME.
    """
    name = name or settings.COLLECTION_NAME
    if name not in _collections:
        _collections[name] = _client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[name]

def drop_collection(name: str):
    _client.delete_collection(name) 
    """Elimina por completo la colección y limpia la caché local."""
    try:
        _client.delete_collection(name)
    except Exception:
        pass  # si no existe, ignoramos
    _collections.pop(name, None)