import os
import tempfile
from pathlib import Path
from minio import Minio
from typing import List
from dotenv import load_dotenv

load_dotenv()  # Carga las variables del .env automáticamente

class MinIOHandler:
    """Clase para interactuar con MinIO y descargar PDFs para indexación."""

    def __init__(self, bucket_name: str = "data"):
        self.bucket_name = bucket_name

        # Obtener endpoint desde variables de entorno
        endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT")
        access_key = os.getenv("MINIO_ROOT_USER")
        secret_key = os.getenv("MINIO_ROOT_PASSWORD")

        if not endpoint or not access_key or not secret_key:
            raise ValueError("Las variables de entorno de MinIO no están definidas correctamente.")

        # Quitar esquema (http:// o https://) si está presente
        if "://" in endpoint:
            scheme, endpoint = endpoint.split("://")
            secure = scheme == "https"
        else:
            secure = True  # Por defecto True si no se indica

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def list_pdfs(self, bucket_name: str = None) -> List[str]:
        bucket_name = bucket_name or self.bucket_name
        pdfs = [
            obj.object_name
            for obj in self.client.list_objects(bucket_name)
            if obj.object_name.lower().endswith(".pdf")
        ]
        return pdfs


    def download_pdf(self, object_name: str, bucket_name: str = None) -> Path:
        bucket_name = bucket_name or self.bucket_name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.close()  # importante para Windows
        self.client.fget_object(bucket_name, object_name, temp_file.name)
        return Path(temp_file.name)


    def download_all_pdfs(self) -> List[Path]:
        """Descarga todos los PDFs del bucket temporalmente."""
        pdf_paths = []
        for pdf_name in self.list_pdfs():
            path = self.download_pdf(pdf_name)
            pdf_paths.append(path)
        return pdf_paths
