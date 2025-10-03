from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable
from minio import Minio

def _endpoint_and_secure() -> tuple[str, bool]:
    ep = os.getenv("MINIO_PRIVATE_ENDPOINT") or os.getenv("MINIO_PUBLIC_ENDPOINT")
    if not ep:
        raise RuntimeError("MINIO_PRIVATE_ENDPOINT or MINIO_PUBLIC_ENDPOINT must be set.")
    if "://" in ep:
        scheme, rest = ep.split("://", 1)
        return rest, scheme == "https"
    return ep, ep.endswith(":443")

def _client() -> Minio:
    endpoint, secure = _endpoint_and_secure()
    user = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY")
    pwd  = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY")
    if not user or not pwd:
        raise RuntimeError("MINIO credentials not set (MINIO_ROOT_USER / MINIO_ROOT_PASSWORD).")
    return Minio(endpoint, access_key=user, secret_key=pwd, secure=secure)

def list_objects(bucket: str, prefix: str) -> Iterable[str]:
    cli = _client()
    for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
        key = obj.object_name
        if not key.endswith("/"):
            yield key

def download_prefix(bucket: str, prefix: str, dest_dir: str) -> None:
    cli = _client()
    base = Path(dest_dir)
    base.mkdir(parents=True, exist_ok=True)
    for key in list_objects(bucket, prefix):
        rel = key[len(prefix):] if key.startswith(prefix) else key
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cli.fget_object(bucket, key, str(path))
