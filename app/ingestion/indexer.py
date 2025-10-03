# app/ingestion/indexer.py
# Index a single PDF file using the same components as pipeline.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
import uuid

from app.ingestion.parsers.factory import get_parser
from app.ingestion.chunker import make_chunks
from app.ingestion.embedder import embed_texts


def _open_collection(persist_dir: str, collection_name: str):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, coll


def index_file(path: Path, parser_name: str = "pymupdf") -> Dict:
    assert path.exists() and path.is_file(), f"File not found: {path}"

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "retie_docs")
    _, coll = _open_collection(persist_dir, collection_name)

    # wipe existing entries for this source
    try:
        coll.delete(where={"source": path.name})
    except Exception:
        pass

    parser = get_parser(parser_name)
    pages: List[Tuple[int, str]] = parser.extract_pages(path)
    total_chunks = 0

    for page_num, text in pages:
        chunks = make_chunks(text, page=page_num)
        if not chunks:
            continue

        payloads = [c.text for c in chunks]
        ids = [f"{path.name}-{page_num}-{c.chunk_id}-{uuid.uuid4().hex[:8]}" for c in chunks]
        metadatas = [{"source": path.name, "page": page_num, "chunk_id": c.chunk_id} for c in chunks]

        embs = embed_texts(payloads)
        coll.add(ids=ids, documents=payloads, embeddings=embs, metadatas=metadatas)
        total_chunks += len(chunks)

    return {"file": str(path), "chunks": total_chunks}
