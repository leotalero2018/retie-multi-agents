# app/bootstrap_sync.py
from __future__ import annotations
import os
import stat
import socket
import logging
from pathlib import Path
from typing import Optional, Tuple

LOG_LEVEL = os.getenv("MINIO_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("bootstrap_sync")

def _redact(v: Optional[str]) -> str:
    if not v:
        return "<unset>"
    if len(v) <= 6:
        return "***"
    return v[:3] + "…" + v[-3:]

def _dump_env():
    log.info("[SYNC] ENV → CHROMA_PERSIST_DIR=%s  CHROMA_DB_DIR=%s  COLLECTION_NAME=%s",
             os.getenv("CHROMA_PERSIST_DIR"), os.getenv("CHROMA_DB_DIR"),
             os.getenv("COLLECTION_NAME"))
    log.info("[SYNC] ENV → MINIO_BUCKET_NAME=%s  MINIO_PREFIX=%s",
             os.getenv("MINIO_BUCKET_NAME"), os.getenv("MINIO_PREFIX"))
    log.info("[SYNC] ENV → MINIO_PUBLIC_ENDPOINT=%s  MINIO_ENDPOINT=%s  MINIO_PRIVATE_ENDPOINT=%s",
             os.getenv("MINIO_PUBLIC_ENDPOINT"), os.getenv("MINIO_ENDPOINT"),
             os.getenv("MINIO_PRIVATE_ENDPOINT"))
    log.info("[SYNC] ENV → MINIO_ROOT_USER=%s  MINIO_ROOT_PASSWORD=%s  FORCE_SYNC=%s",
             _redact(os.getenv("MINIO_ROOT_USER")), _redact(os.getenv("MINIO_ROOT_PASSWORD")),
             os.getenv("MINIO_FORCE_SYNC"))

def _dns_check(host_port: str) -> Tuple[bool, str]:
    try:
        host = host_port
        if "://" in host_port:
            host = host_port.split("://", 1)[1]
        if ":" in host:
            host = host.split(":", 1)[0]
        socket.getaddrinfo(host, None)
        return True, f"DNS OK ({host})"
    except Exception as e:
        return False, f"DNS FAIL ({host_port}): {e}"

def _chroma_dir_has_db(dirpath: str) -> bool:
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        return False
    sqlite = p / "chroma.sqlite3"
    has_shard = any(p.glob("*.db"))
    return sqlite.exists() and has_shard

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _select_endpoint() -> Tuple[str, bool]:
    """
    Returns (endpoint_host_port, secure).
    Preference: PUBLIC → ENDPOINT → PRIVATE.
    """
    endpoint = (os.getenv("MINIO_PUBLIC_ENDPOINT")
                or os.getenv("MINIO_ENDPOINT")
                or os.getenv("MINIO_PRIVATE_ENDPOINT"))
    if not endpoint:
        raise RuntimeError("No MINIO_* endpoint set")
    secure = True
    if "://" in endpoint:
        scheme, rest = endpoint.split("://", 1)
        endpoint = rest
        secure = scheme == "https"
    else:
        secure = endpoint.endswith(":443")
    return endpoint, secure

def _minio_client():
    try:
        from minio import Minio
    except Exception as e:
        raise RuntimeError("minio package not installed") from e

    endpoint, secure = _select_endpoint()
    access = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("Missing MINIO credentials")

    ok, msg = _dns_check(endpoint)
    log.info("[SYNC] Endpoint=%s  secure=%s  %s", endpoint, secure, msg)

    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)

def _normalize_prefix(raw: Optional[str], default: str = "chroma_db/") -> str:
    """
    Trim spaces, drop leading slash, and ensure trailing slash.
    Turns ' data/chroma_db ', '/data/chroma_db', 'data/chroma_db'
    into 'data/chroma_db/'.
    """
    s = (raw if raw is not None else default).strip()
    s = s.lstrip("/")
    if s and not s.endswith("/"):
        s += "/"
    return s

def _fix_permissions(root: str):
    """
    Make downloaded files readable/writable by the app user (and SQLite).
    """
    try:
        for dp, dn, fn in os.walk(root):
            os.chmod(dp, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            for f in fn:
                p = os.path.join(dp, f)
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR |
                            stat.S_IRGRP | stat.S_IWGRP |
                            stat.S_IROTH | stat.S_IWOTH)
    except Exception as e:
        log.warning("[SYNC] Could not fix permissions under %s: %s", root, e)

def _download_prefix(minio_cli, bucket: str, prefix: str, local_dir: str) -> int:
    """
    Recursively download all objects under `prefix` into `local_dir`.
    Returns number of objects downloaded.
    """
    _ensure_dir(local_dir)

    # Pre-flight checks
    try:
        exists = minio_cli.bucket_exists(bucket)
        log.info("[SYNC] bucket_exists(%s) → %s", bucket, exists)
        if not exists:
            raise RuntimeError(f"Bucket '{bucket}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"Bucket check failed: {e}")

    # List objects
    objs = list(minio_cli.list_objects(bucket, prefix=prefix, recursive=True))
    log.info("[SYNC] list_objects bucket=%s prefix='%s' → %d objects", bucket, prefix, len(objs))

    # Show first few keys for validation
    for i, obj in enumerate(objs[:20], 1):
        log.info("[SYNC]   %2d) %s  (%d bytes)", i, obj.object_name, getattr(obj, "size", -1))

    if not objs:
        raise RuntimeError(f"Bucket '{bucket}' has no objects under prefix '{prefix}'")

    # Download
    downloaded = 0
    for obj in objs:
        rel_key = obj.object_name[len(prefix):] if prefix else obj.object_name
        dest = Path(local_dir) / rel_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        minio_cli.fget_object(bucket, obj.object_name, str(dest))
        downloaded += 1
    return downloaded

def sync_chroma_from_minio() -> None:
    """
    Try MinIO first. On any error, log and fall back to local files.
    """
    _dump_env()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR") or "./data/chroma_db"
    bucket = os.getenv("MINIO_BUCKET_NAME")
    prefix = _normalize_prefix(os.getenv("MINIO_PREFIX"), default="chroma_db/")
    force = os.getenv("MINIO_FORCE_SYNC", "false").lower() in ("1", "true", "yes")

    _ensure_dir(persist_dir)

    if _chroma_dir_has_db(persist_dir) and not force:
        log.info("[SYNC] Local Chroma dir '%s' already complete; skipping download.", persist_dir)
        return

    if not bucket:
        log.info("[SYNC] MINIO_BUCKET_NAME not set; skipping MinIO sync.")
        return

    try:
        client = _minio_client()
    except Exception as e:
        log.error("[SYNC] MinIO client unavailable: %s", e)
        return

    try:
        log.info("[SYNC] Descargando minio://%s/%s -> %s (force=%s)", bucket, prefix, persist_dir, force)
        count = _download_prefix(client, bucket=bucket, prefix=prefix, local_dir=persist_dir)
        log.info("[SYNC] Descarga completa. Objetos descargados: %d", count)

        # Ensure the DB is writable for SQLite/HNSW
        _fix_permissions(persist_dir)

        # Verify markers
        sqlite_ok = (Path(persist_dir) / "chroma.sqlite3").is_file()
        shards = [d for d in Path(persist_dir).iterdir() if d.is_dir() and d.name.endswith(".db")]
        if sqlite_ok and shards:
            log.info("[SYNC] Verificación OK: sqlite y shard(s) presentes: %s", [s.name for s in shards])
        else:
            log.warning("[SYNC] Descargado, pero faltan marcadores de DB (sqlite=%s, shards=%s).",
                        sqlite_ok, bool(shards))

    except Exception as e:
        log.error("[SYNC] Falló la descarga: %s", e)
        # Fall-through: app continues with whatever is local
