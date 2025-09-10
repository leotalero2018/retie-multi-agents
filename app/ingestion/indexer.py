from pathlib import Path
from typing import Dict, List
from uuid import uuid4
import hashlib 

from app.ingestion.extractors import (
    extract_text_from_pdf,
    extract_text_from_pptx,
    extract_text_from_image,
)
from app.ingestion.chunker import make_chunks
from app.ingestion.embedder import embed_texts
from app.retriever.chroma_client import get_collection
from app.ingestion.parsers.factory import get_parser


SUPPORTED = {".pdf", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".txt",".md",".docx"}


def _extract_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in {".ppt", ".pptx"}:
        return extract_text_from_pptx(path)
    if ext in {".png", ".jpg", ".jpeg"}:
        return extract_text_from_image(path)
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Extensión no soportada: {ext}")


def index_file(path: Path, ocr_lang: str = "spa+eng", collection_name: str | None = None, parser_name: str | None = None) -> Dict:
    assert path.exists() and path.is_file()

    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    parser = get_parser(parser_name or "pdfplumber")
    if path.suffix.lower() == ".pdf":
        pages = parser.extract_pages(path)  # parser controla cómo extrae el PDF
    else:
        # Reuso de tu flujo existente para PPT/IMG/TXT si quieres;
        # aquí simplificamos: tratamos todo por parser; si no es PDF, no procesar.
        # (Puedes extender para otros formatos igual)
        raise ValueError("Por ahora este flujo solo trata PDFs con parsers seleccionables.")

    collection = get_collection(collection_name)
    collection.delete(where={"source": path.name})

    total_chunks = 0
    ids: List[str] = []
    payloads: List[str] = []
    metadatas: List[dict] = []

    for page_num, text in pages:
        chunks = make_chunks(text, page=page_num)
        for c in chunks:
            total_chunks += 1
            ids.append(f"{sha}-{page_num}-{c.chunk_id}")
            payloads.append(c.text)
            metadatas.append(
                {"source": path.name, "page": page_num, "chunk_id": c.chunk_id, "file_sha256": sha}
            )

    if not payloads:
        return {"file": str(path), "chunks": 0}

    embs = embed_texts(payloads)
    collection.add(ids=ids, documents=payloads, embeddings=embs, metadatas=metadatas)
    return {"file": str(path), "chunks": total_chunks}