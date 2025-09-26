# app/api/debug.py
import os
import urllib.parse
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()

# Intentaremos usar tu MinIOHandler si existe; si no, caemos a una conexión directa.
try:
    from app.ingestion.minio_handler import MinIOHandler  # nombre que creaste
except Exception:
    MinIOHandler = None

@router.get("/debug/minio")
def debug_minio(secret: str | None = None):
    """
    Endpoint de debug que lista los PDFs dentro del bucket MINIO_BUCKET.
    Opcionalmente protegido por MINIO_DEBUG_SECRET (valor en env).
    """
    # --- seguridad opcional ---
    debug_secret = os.getenv("MINIO_DEBUG_SECRET")
    if debug_secret:
        if secret != debug_secret:
            raise HTTPException(status_code=403, detail="Forbidden")

    bucket = os.getenv("MINIO_BUCKET", "data")

    try:
        # 1) Si existe tu clase MinIOHandler, úsala
        if MinIOHandler:
            handler = MinIOHandler(bucket_name=bucket)
            pdfs: List[str] = handler.list_pdfs()
            return {"ok": True, "pdfs": pdfs}

        # 2) Si no existe, creamos cliente Minio "manualmente"
        from minio import Minio

        endpoint = os.getenv("MINIO_ENDPOINT") or os.getenv("MINIO_PUBLIC_ENDPOINT")
        if not endpoint:
            raise RuntimeError("No MINIO_ENDPOINT found in environment")

        # Si endpoint tiene esquema, extraemos host:port para Minio constructor
        if endpoint.startswith("http"):
            parsed = urllib.parse.urlparse(endpoint)
            endpoint_host = parsed.netloc
            # preserve scheme for secure
            secure = parsed.scheme == "https"
        else:
            endpoint_host = endpoint
            secure = True

        client = Minio(
            endpoint_host,
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=secure,
        )

        pdfs = [obj.object_name for obj in client.list_objects(bucket)]
        return {"ok": True, "pdfs": pdfs}

    except Exception as e:
        return {"ok": False, "error": str(e)}
