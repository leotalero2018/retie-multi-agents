"""E6 — Caché persistente de embeddings (SQLite).

Clave = sha256(modelo + texto). Una re-corrida del pipeline solo paga los
embeddings de chunks cuyo texto cambió; el resto sale del caché local.
El archivo vive en v2_work/ (NO se publica).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from array import array
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, path: Path | str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, model TEXT, vec BLOB)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8", errors="ignore")).hexdigest()

    def get_many(self, keys: List[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        CHUNK = 500
        for i in range(0, len(keys), CHUNK):
            batch = keys[i:i + CHUNK]
            q = f"SELECT k, vec FROM emb WHERE k IN ({','.join('?' * len(batch))})"
            for k, blob in self._conn.execute(q, batch):
                a = array("f")
                a.frombytes(blob)
                out[k] = list(a)
        return out

    def put_many(self, items: Dict[str, List[float]], model: str) -> None:
        rows = [(k, model, array("f", v).tobytes()) for k, v in items.items()]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO emb (k, model, vec) VALUES (?, ?, ?)", rows)

    def close(self) -> None:
        self._conn.close()


def embed_with_cache(texts: List[str], cache: EmbeddingCache, *,
                     batch_size: int = 100) -> List[List[float]]:
    """Embebe una lista de textos usando el caché. Los misses se calculan con
    el MISMO embedder del agente (retie_agent.llm.embedder) en batches."""
    from retie_agent.llm.embedder import embed_texts
    from retie_agent.config import settings

    model = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
    keys = [EmbeddingCache.key(model, t) for t in texts]
    cached = cache.get_many(list(set(keys)))

    miss_idx = [i for i, k in enumerate(keys) if k not in cached]
    if miss_idx:
        log.info("[E6] embeddings: %d en caché, %d por calcular",
                 len(texts) - len(miss_idx), len(miss_idx))
    new: Dict[str, List[float]] = {}
    for start in range(0, len(miss_idx), batch_size):
        idx_batch = miss_idx[start:start + batch_size]
        vecs = embed_texts([texts[i] for i in idx_batch])
        for i, v in zip(idx_batch, vecs):
            new[keys[i]] = v
        if start // batch_size % 5 == 0 and miss_idx:
            log.info("[E6]   … %d/%d embebidos", min(start + batch_size, len(miss_idx)), len(miss_idx))
    if new:
        cache.put_many(new, model)

    merged = {**cached, **new}
    return [merged[k] for k in keys]
