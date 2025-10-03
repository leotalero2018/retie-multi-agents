from __future__ import annotations
import os
from pathlib import Path

def _is_chroma_ready(dest: str) -> bool:
    """
    Considerar la BD 'lista' solo si existen archivos clave (no cualquier cosa).
    """
    p = Path(dest)
    if not p.exists():
        return False
    # chroma.sqlite3 es el marcador principal
    sqlite = p / "chroma.sqlite3"
    if sqlite.exists() and sqlite.is_file():
        return True
    # En layouts nuevos de Chroma puede no haber sqlite, pero sí shards (*db)
    shards = list(p.glob("*db"))
    return bool(shards)

def sync_chroma_from_minio() -> None:
    """
    Descargar todos los objetos de MinIO al directorio CHROMA_PERSIST_DIR si no
    está 'listo', o si MINIO_FORCE_SYNC=true.
    """
    dest   = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    bucket = os.getenv("MINIO_BUCKET_NAME") or os.getenv("MINIO_BUCKET_FILE")
    prefix = os.getenv("MINIO_PREFIX", "chroma_db/")
    force  = (os.getenv("MINIO_FORCE_SYNC", "false").lower() in ("1","true","yes"))

    p = Path(dest)
    p.mkdir(parents=True, exist_ok=True)

    if not bucket:
        print("[SYNC] MINIO_BUCKET_NAME not set; skipping MinIO sync.")
        return

    if not force and _is_chroma_ready(dest):
        print(f"[SYNC] Chroma dir '{dest}' already has chroma.sqlite3 (ready); skipping download.")
        return

    print(f"[SYNC] Downloading minio://{bucket}/{prefix} -> {dest} (force={force})")
    try:
        from app.services.storage_minio import download_prefix
        download_prefix(bucket=bucket, prefix=prefix, dest_dir=dest)
        print("[SYNC] Done.")
    except Exception as e:
        print(f"[SYNC] ERROR: {e}")
        # Si la BD es obligatoria, puedes forzar fallo:
        # import sys; sys.exit(1)
