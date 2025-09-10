# app/utils/storage.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional
from minio import Minio
from minio.error import S3Error
from app.config import settings

_client: Optional[Minio] = None

def get_minio() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _client

def ensure_bucket(bucket: str):
    cli = get_minio()
    found = cli.bucket_exists(bucket)
    if not found:
        cli.make_bucket(bucket)

def put_file(bucket: str, local_path: Path, object_name: Optional[str] = None) -> str:
    ensure_bucket(bucket)
    cli = get_minio()

    object_name = object_name or local_path.name
    cli.fput_object(bucket, object_name, str(local_path))
    # URL presignada temporal (ej.: 7 días)
    url = cli.presigned_get_object(bucket, object_name)
    return url

def build_object_name(chat_id: int, prefix: str, suffix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}/{chat_id}/{ts}.{suffix}"
