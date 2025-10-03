# app/ingestion/pipeline.py
# Single entrypoint to index a folder of PDFs into Chroma using chosen parser and explicit embeddings.

from __future__ import annotations

import os
import uuid
import pathlib
from typing import List, Dict, Tuple

def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

# Local deps
from app.ingestion.parsers.factory import get_parser
from app.ingestion.chunker import make_chunks
from app.ingestion.embedder import embed_texts


def _open_collection(persist_dir: str, collection_name: str):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, coll


def _index_pdf(path: pathlib.Path, coll, parser_name: str) -> int:
    parser = get_parser(parser_name)
    pages: List[Tuple[int, str]] = parser.extract_pages(path)  # [(page_num, text)]
    if not pages:
        return 0

    total = 0
    for page_num, text in pages:
        chunks = make_chunks(text, page=page_num)
        if not chunks:
            continue
        docs = [c.text for c in chunks]
        ids = [f"{path.name}-{page_num}-{c.chunk_id}-{uuid.uuid4().hex[:8]}" for c in chunks]
        metas = [{"source": path.name, "page": page_num, "chunk_id": c.chunk_id} for c in chunks]

        # Compute embeddings explicitly (no need to bind an embedding fn to Chroma)
        embs = embed_texts(docs)
        coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        total += len(chunks)
    return total


def index_folder(source_dir: str, persist_dir: str, engine: str = "pymupdf") -> int:
    """
    Index all PDFs in source_dir into the Chroma collection named by COLLECTION_NAME.
    Returns number of chunks written.
    """
    collection_name = os.getenv("COLLECTION_NAME", "retie_docs")
    client, coll = _open_collection(persist_dir=persist_dir, collection_name=collection_name)

    total_chunks = 0
    source = pathlib.Path(source_dir)
    for pdf in sorted(source.glob("*.pdf")):
        # wipe previous content for this specific source (idempotent per file)
        try:
            coll.delete(where={"source": pdf.name})
        except Exception:
            pass
        total_chunks += _index_pdf(pdf, coll, parser_name=engine)

    return total_chunks
