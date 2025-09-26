# app/ingestion/index_minio.py
import os
import logging
from pathlib import Path
from typing import Dict, List
from app.ingestion.minio_handler import MinIOHandler
from app.ingestion.extractors import extract_pages
from app.ingestion.chunker import make_chunks
from app.ingestion.embedder import embed_texts
from app.retriever.chroma_client import get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("index_minio")

def ingest_from_minio(bucket_name: str | None = None, collection_name: str | None = None, rebuild: bool = False) -> Dict:
    """
    Extrae PDFs desde MinIO, divide en chunks, genera embeddings y los guarda en la colección Chroma.
    - bucket_name: nombre del bucket en MinIO (por defecto toma MINIO_BUCKET)
    - collection_name: nombre de la colección Chroma (por defecto toma COLLECTION_NAME)
    - rebuild: si True, borra los documentos con el mismo 'source' antes de reindexar por archivo.
    Retorna un resumen con totales.
    """
    bucket = bucket_name or os.getenv("MINIO_BUCKET", "data")
    coll_name = collection_name or os.getenv("COLLECTION_NAME", "retie_docs")

    logger.info("Ingest: bucket=%s, collection=%s, rebuild=%s", bucket, coll_name, rebuild)

    # 1) conectar a MinIO
    minio = MinIOHandler(bucket_name=bucket)

    # 2) listar PDFs
    pdfs: List[str] = minio.list_pdfs()
    if not pdfs:
        logger.info("No se encontraron PDFs en el bucket '%s'.", bucket)
        return {"ok": True, "processed": 0, "message": "No PDFs found."}

    # 3) obtener colección (Chroma)
    collection = get_collection(coll_name)

    total_chunks = 0
    processed_files = []

    for pdf_name in pdfs:
        logger.info("Procesando PDF: %s", pdf_name)

        # (opcional) eliminar contenido previo de esta fuente
        if rebuild:
            try:
                collection.delete(where={"source": pdf_name})
                logger.info("Se eliminaron entradas previas para source=%s", pdf_name)
            except Exception as e:
                logger.warning("No se pudo eliminar entradas previas para %s: %s", pdf_name, e)

        try:
            # 4) descargar temporalmente
            local_path = minio.download_pdf(pdf_name)
            logger.info("Descargado temporal: %s", local_path)

            # 5) extraer páginas (lista de tuplas (page_num, text))
            pages = extract_pages(Path(local_path))
            logger.info("Páginas extraídas: %d", len(pages))

            # 6) por cada página: chunk -> embeddings -> agregar a collection
            for page_num, text in pages:
                chunks = make_chunks(text, page=page_num)
                if not chunks:
                    continue

                payloads = [c.text for c in chunks]
                ids = [f"{pdf_name}-{page_num}-{c.chunk_id}" for c in chunks]
                metadatas = [{"source": pdf_name, "page": page_num, "chunk_id": c.chunk_id} for c in chunks]

                # generar embeddings (usa el provider configurado en app.ingestion.embedder)
                embeddings = embed_texts(payloads)

                # agregar a Chroma (o la DB vectorial que uses actualmente)
                collection.add(ids=ids, documents=payloads, embeddings=embeddings, metadatas=metadatas)

                total_chunks += len(chunks)

            processed_files.append(pdf_name)

        except Exception as e:
            logger.exception("Error procesando %s: %s", pdf_name, e)
        finally:
            # 7) limpiar archivo temporal si existe
            try:
                Path(local_path).unlink(missing_ok=True)
            except Exception:
                pass

    logger.info("Ingest completo. archivos=%d, chunks=%d", len(processed_files), total_chunks)
    return {"ok": True, "processed_files": processed_files, "total_chunks": total_chunks}

# Permitir ejecución como script para pruebas locales:
if __name__ == "__main__":
    # Puedes pasar variables por env o usar valores por defecto
    res = ingest_from_minio()
    print(res)
