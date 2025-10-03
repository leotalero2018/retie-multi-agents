# app/bootstrap_sync.py
from __future__ import annotations
import os
from pathlib import Path

def sync_chroma_from_minio() -> None:
    """
    If CHROMA_PERSIST_DIR is empty, download the DB from MinIO:
      - MINIO_BUCKET_NAME (your bucket)
      - MINIO_PREFIX (defaults to 'chroma_db/')
    """
    dest = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    bucket = os.getenv("MINIO_BUCKET_NAME") or os.getenv("MINIO_BUCKET_FILE")  # you have both; NAME is primary
    prefix = os.getenv("MINIO_PREFIX", "chroma_db/")

    if not bucket:
        print("[SYNC] MINIO_BUCKET_NAME not set; skipping MinIO sync.")
        return

    p = Path(dest)
    p.mkdir(parents=True, exist_ok=True)

    # Skip download if already populated (fast startup)
    if any(p.iterdir()):
        print(f"[SYNC] Local Chroma dir '{dest}' already populated; skipping download.")
        return

    print(f"[SYNC] Downloading minio://{bucket}/{prefix} -> {dest}")
    try:
        from app.services.storage_minio import download_prefix
        download_prefix(bucket=bucket, prefix=prefix, dest_dir=dest)
        print("[SYNC] Done.")
    except Exception as e:
        print(f"[SYNC] ERROR: {e}")
        # If your service MUST have the DB, you can hard-fail here:
        # import sys; sys.exit(1)
