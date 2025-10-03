# app/services/storage_minio.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from minio import Minio

def _endpoint_and_secure() -> tuple[str, bool]:
    """
    Prefer the internal endpoint inside Railway network (faster, no TLS).
    Fallback to the public endpoint (TLS).
    Accepts both with/without scheme.
    """
    ep = os.getenv("MINIO_PRIVATE_ENDPOINT") or os.getenv("MINIO_PUBLIC_ENDPOINT")
    if not ep:
        raise RuntimeError("MINIO_PRIVATE_ENDPOINT or MINIO_PUBLIC_ENDPOINT must be set.")

    if "://" in ep:
        scheme, rest = ep.split("://", 1)
        return rest, scheme == "https"
    # Heuristic: internal endpoint likely non-TLS; public 443 is TLS.
    return ep, ep.endswith(":443")

def _client() -> Minio:
    endpoint, secure = _endpoint_and_secure()
    user = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USERNAME")
    pwd  = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    if not user or not pwd:
        raise RuntimeError("MINIO credentials not set (MINIO_ROOT_USER / MINIO_ROOT_PASSWORD).")
    return Minio(endpoint, access_key=user, secret_key=pwd, secure=secure)

def list_objects(bucket: str, prefix: str) -> Iterable[str]:
    cli = _client()
    for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
        key = obj.object_name
        if key.endswith("/"):
            continue
        yield key

def download_prefix(bucket: str, prefix: str, dest_dir: str) -> None:
    """
    Download all objects under s3://bucket/prefix to dest_dir.
    Keeps subfolder structure (prefix is stripped from local paths).
    """
    cli = _client()
    base = Path(dest_dir)
    base.mkdir(parents=True, exist_ok=True)

    for key in list_objects(bucket, prefix):
        rel = key[len(prefix):] if key.startswith(prefix) else key
        local_path = base / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cli.fget_object(bucket, key, str(local_path))
