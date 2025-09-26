# test_minio.py
from dotenv import load_dotenv
from app.ingestion.minio_handler import MinIOHandler

from dotenv import load_dotenv
import os

load_dotenv()  # carga automáticamente las variables del archivo .env

print("Endpoint:", os.getenv("MINIO_PUBLIC_ENDPOINT"))
print("User:", os.getenv("MINIO_ROOT_USER"))


load_dotenv()

# Crear instancia
minio = MinIOHandler(bucket_name="data")

# Listar PDFs
pdf_list = minio.list_pdfs()
print("PDFs en MinIO:", pdf_list)

# Descargar todos temporalmente
pdf_paths = minio.download_all_pdfs()
for path in pdf_paths:
    print("Descargado:", path)
