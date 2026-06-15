"""Construcción del índice ChromaDB final (out_dir) a partir de los chunks.

El índice se RECONSTRUYE desde cero en cada corrida (barato gracias al caché
de embeddings E6) — evita estados mezclados. Los assets y el registro se
copian dentro de out_dir para que TODO se publique como una unidad.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List

from .chunking import ChunkDoc
from .config import V2Config
from .embed_cache import EmbeddingCache, embed_with_cache

log = logging.getLogger(__name__)


def build_index(cfg: V2Config, all_chunks: List[ChunkDoc],
                doc_summaries: Dict[str, dict]) -> Dict[str, int]:
    """Crea out_dir desde cero: Chroma + assets + registro + manifest."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if cfg.out_dir.exists():
        shutil.rmtree(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True)

    client = chromadb.PersistentClient(
        path=str(cfg.out_dir), settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_or_create_collection(
        name=cfg.collection, metadata={"hnsw:space": "cosine"})

    cache = EmbeddingCache(cfg.embcache_path)
    texts = [c.text for c in all_chunks]
    t0 = time.time()
    embeddings = embed_with_cache(texts, cache, batch_size=cfg.embed_batch)
    cache.close()
    log.info("[INDEX] %d embeddings listos en %.1fs", len(embeddings), time.time() - t0)

    BATCH = 500
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start:start + BATCH]
        col.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embeddings[start:start + len(batch)],
            metadatas=[_sanitize_meta(c.metadata) for c in batch],
        )
    count = col.count()
    log.info("[INDEX] colección '%s': %d chunks", cfg.collection, count)

    # Copiar assets + registro dentro del índice (publicación como unidad)
    if cfg.assets_dir.exists():
        shutil.copytree(cfg.assets_dir, cfg.out_dir / "assets")
    if cfg.registry_path.exists():
        shutil.copy2(cfg.registry_path, cfg.out_dir / "assets_registry.sqlite")

    # Manifest
    from retie_agent.config import settings as agent_settings
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": cfg.schema_version,
        "collection": cfg.collection,
        "embedding_model": getattr(agent_settings, "EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dim": len(embeddings[0]) if embeddings else None,
        "chunks": count,
        "docs": doc_summaries,
    }
    (cfg.out_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"chunks": count}


def _sanitize_meta(meta: dict) -> dict:
    """Chroma solo acepta primitivos en metadata."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            continue
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out
