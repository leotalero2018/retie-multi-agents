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

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _select_endpoint() -> Tuple[str, bool]:
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
    s = (raw if raw is not None else default).strip()
    s = s.lstrip("/")
    if s and not s.endswith("/"):
        s += "/"
    return s

def _fix_permissions(root: str):
    try:
        for dp, dn, fn in os.walk(root):
            os.chmod(dp, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)     # 777 for dirs
            for f in fn:
                p = os.path.join(dp, f)
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR |               # 666 for files
                             stat.S_IRGRP | stat.S_IWGRP |
                             stat.S_IROTH | stat.S_IWOTH)
    except Exception as e:
        log.warning("[SYNC] Could not fix permissions under %s: %s", root, e)

def _rw_sanity(dirpath: str) -> Tuple[bool, str]:
    """Try to create + delete a tiny temp file; report reason if it fails."""
    p = Path(dirpath)
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".rw_test"
        with open(test, "w") as fh:
            fh.write("ok")
        test.unlink()
        return True, "RW OK"
    except Exception as e:
        return False, f"RW FAIL: {e}"

def _markers(dirpath: str) -> Tuple[bool, bool, list[str]]:
    """Return (sqlite_present, shards_present, shard_names). Shards = ANY directory (UUID or *.db)."""
    p = Path(dirpath)
    sqlite_ok = (p / "chroma.sqlite3").is_file()
    shard_dirs = [d.name for d in p.iterdir() if d.is_dir()] if p.exists() else []
    # accept any directory name (UUID or UUID.db)
    shards_ok = len(shard_dirs) > 0
    return sqlite_ok, shards_ok, shard_dirs

def _download_prefix(minio_cli, bucket: str, prefix: str, local_dir: str) -> int:
    _ensure_dir(local_dir)

    try:
        exists = minio_cli.bucket_exists(bucket)
        log.info("[SYNC] bucket_exists(%s) → %s", bucket, exists)
        if not exists:
            raise RuntimeError(f"Bucket '{bucket}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"Bucket check failed: {e}")

    objs = list(minio_cli.list_objects(bucket, prefix=prefix, recursive=True))
    log.info("[SYNC] list_objects bucket=%s prefix='%s' → %d objects", bucket, prefix, len(objs))
    for i, obj in enumerate(objs[:20], 1):
        log.info("[SYNC]   %2d) %s  (%d bytes)", i, obj.object_name, getattr(obj, "size", -1))

    if not objs:
        raise RuntimeError(f"Bucket '{bucket}' has no objects under prefix '{prefix}'")

    downloaded = 0
    for obj in objs:
        rel_key = obj.object_name[len(prefix):] if prefix else obj.object_name
        dest = Path(local_dir) / rel_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        minio_cli.fget_object(bucket, obj.object_name, str(dest))
        downloaded += 1
    return downloaded

def sync_chroma_from_minio() -> None:
    _dump_env()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR") or "./data/chroma_db"
    bucket = os.getenv("MINIO_BUCKET_NAME")
    prefix = _normalize_prefix(os.getenv("MINIO_PREFIX"), default="chroma_db/")
    force = os.getenv("MINIO_FORCE_SYNC", "false").lower() in ("1", "true", "yes")

    _ensure_dir(persist_dir)

    # If already complete and not forced, skip
    sqlite_ok, shards_ok, shard_names = _markers(persist_dir)
    if sqlite_ok and shards_ok and not force:
        log.info("[SYNC] Local dir '%s' already complete; skipping download (shards=%s).",
                 persist_dir, shard_names)
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

        # perms and RW test
        _fix_permissions(persist_dir)
        ok, msg = _rw_sanity(persist_dir)
        log.info("[SYNC] RW check for %s → %s", persist_dir, msg)

        sqlite_ok, shards_ok, shard_names = _markers(persist_dir)
        if sqlite_ok and shards_ok:
            log.info("[SYNC] Verificación OK: sqlite y shard(s) presentes: %s", shard_names)
        else:
            log.warning("[SYNC] Descargado, pero faltan marcadores de DB (sqlite=%s, shards=%s, found=%s).",
                        sqlite_ok, shards_ok, shard_names)

    except Exception as e:
        log.error("[SYNC] Falló la descarga: %s", e)
        # Fall-through to local
