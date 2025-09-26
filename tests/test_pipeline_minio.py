# tests/test_pipeline_minio.py
import tempfile
from pathlib import Path

from app.ingestion.minio_handler import MinIOHandler
from app.ingestion.extractors import extract_pages
from app.ingestion.chunker import make_chunks
from app.ingestion.embedder import embed_texts
from app.retriever.chroma_client import get_collection

def test_pipeline_minio():
    # 1. Inicializamos el cliente MinIO
    handler = MinIOHandler()
    bucket = "data"

    # 2. Listamos PDFs disponibles
    pdfs = handler.list_pdfs(bucket)
    pdfs = handler.list_pdfs()
    print(f"📂 PDFs en MinIO: {pdfs}")
    assert pdfs, "No hay PDFs en el bucket."

    # 3. Descargamos un PDF temporal
    #tmp_path = handler.download_pdf(bucket, pdfs[0])
    tmp_path = handler.download_pdf(pdfs[0], bucket)  # si quieres bucket dinámico
    tmp_path = handler.download_pdf(pdfs[0])          # usa el bucket por defecto

    print(f"📥 Descargado: {tmp_path}")

    # 4. Extraemos páginas de texto
    pages = extract_pages(Path(tmp_path))
    print(f"📑 Páginas extraídas: {len(pages)}")
    assert pages, "No se extrajo texto del PDF."

    # 5. Chunkear texto
    chunks = []
    for page_num, text in pages:
        chunks.extend(make_chunks(text, page=page_num))
    print(f"🔖 Chunks creados: {len(chunks)}")
    assert chunks, "No se generaron chunks."

    # 6. Generar embeddings
    texts = [c.text for c in chunks]
    embs = embed_texts(texts)
    print(f"📊 Embeddings generados: {len(embs)}")
    assert embs, "No se generaron embeddings."

    # 7. Guardar en Chroma
    col = get_collection("minio_test")
    ids = [f"{pdfs[0]}-p{c.page}-{c.chunk_id}" for c in chunks]
    metadatas = [{"source": pdfs[0], "page": c.page, "chunk_id": c.chunk_id} for c in chunks]

    col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metadatas)
    print(f"✅ Guardados {len(chunks)} chunks en colección 'minio_test'")

if __name__ == "__main__":
    test_pipeline_minio()
