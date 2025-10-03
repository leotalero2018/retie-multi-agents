from __future__ import annotations
import os
from pathlib import Path

def sync_chroma_from_minio() -> None:
    dest   = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    bucket = os.getenv("MINIO_BUCKET_NAME") or os.getenv("MINIO_BUCKET_FILE")
    prefix = os.getenv("MINIO_PREFIX", "chroma_db/")

    if not bucket:
        print("[SYNC] MINIO_BUCKET_NAME not set; skipping MinIO sync.")
        return

    p = Path(dest)
    p.mkdir(parents=True, exist_ok=True)

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
        # Si la DB es obligatoria, descomenta:
        # import sys; sys.exit(1)
