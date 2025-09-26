# app/ingestion/index_minio.py

import os
import tempfile
from app.ingestion.minio_handler import MinioHandler
from app.ingestion.pipeline import process_pdfs_and_store

def main():
    # 🔑 Configuración desde variables de entorno (.env o Railway)
    endpoint = os.getenv("MINIO_ENDPOINT")
    user = os.getenv("MINIO_ROOT_USER")
    password = os.getenv("MINIO_ROOT_PASSWORD")
    bucket = os.getenv("MINIO_BUCKET", "data")

    # 📂 Inicializar handler de MinIO
    minio = MinioHandler(endpoint, user, password, bucket)

    # 📝 Listar PDFs disponibles
    pdfs = minio.list_pdfs()
    print(f"📂 PDFs encontrados en MinIO: {pdfs}")

    # ⬇️ Descargar y procesar
    for pdf in pdfs:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            minio.download_pdf(pdf, tmp.name)
            print(f"📥 Descargado: {tmp.name}")

            # 🧩 Procesar e indexar en Chroma/pgvector
            process_pdfs_and_store([tmp.name], collection_name="docs_collection")

if __name__ == "__main__":
    main()
