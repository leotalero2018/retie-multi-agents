# app/bootstrap_sync.py
from __future__ import annotations
import os
import stat
import socket
import logging
import shutil
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
            os.chmod(dp, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)     # 777 dirs
            for f in fn:
                p = os.path.join(dp, f)
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR |
                             stat.S_IRGRP | stat.S_IWGRP |
                             stat.S_IROTH | stat.S_IWOTH)                # 666 files
    except Exception as e:
        log.warning("[SYNC] Could not fix permissions under %s: %s", root, e)

def _markers(dirpath: str) -> Tuple[bool, bool, list[str]]:
    p = Path(dirpath)
    sqlite_ok = (p / "chroma.sqlite3").is_file()
    shard_dirs = [d.name for d in p.iterdir() if d.is_dir()] if p.exists() else []
    shards_ok = len(shard_dirs) > 0
    return sqlite_ok, shards_ok, shard_dirs


def _local_db_usable(dirpath: str) -> bool:
    """Valida que la DB local tenga las colecciones esperadas Y la dimensión de
    embeddings correcta. Un volumen con una DB vieja (otras colecciones, u otro
    modelo de embeddings) pasaba el check de markers y dejaba el bot sin datos
    útiles aunque MinIO ya tuviera el índice bueno."""
    expected = [
        n.strip() for n in
        (os.getenv("COLLECTION_NAMES") or "normativas").split(",")
        if n.strip()
    ]
    try:
        expected_dim = int(os.getenv("EXPECTED_EMBEDDING_DIM", "1536"))
    except ValueError:
        expected_dim = 1536
    try:
        import chromadb
        cli = chromadb.PersistentClient(path=dirpath)
        names = {c.name for c in cli.list_collections()}
        present = [n for n in expected if n in names]
        if not present:
            log.warning("[SYNC] DB local sin colecciones esperadas %s (tiene %s)",
                        expected, sorted(names))
            return False
        col = cli.get_collection(present[0])
        peek = col.peek(1)
        embs = peek.get("embeddings")
        if embs is not None and len(embs) > 0:
            dim = len(embs[0])
            if dim != expected_dim:
                log.warning("[SYNC] DB local con embeddings de %d dims (se esperan %d) — se re-descarga",
                            dim, expected_dim)
                return False
        return True
    except Exception as e:
        log.warning("[SYNC] No se pudo validar la DB local (%s); se asume utilizable", e)
        return True

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

def sync_chroma_from_minio() -> str:
    """
    Sync from MinIO into CHROMA_PERSIST_DIR, then copy into a guaranteed writable
    runtime dir (MINIO_RUNTIME_DIR or /tmp/chroma_db). Return the final dir path
    you should point Chroma to.
    """
    _dump_env()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR") or "./data/chroma_db"
    # Default runtime_dir to persist_dir so we never delete what we just downloaded
    runtime_dir = os.getenv("MINIO_RUNTIME_DIR") or persist_dir
    bucket = os.getenv("MINIO_BUCKET_NAME")
    prefix = _normalize_prefix(os.getenv("MINIO_PREFIX"), default="chroma_db/")
    force = os.getenv("MINIO_FORCE_SYNC", "false").lower() in ("1", "true", "yes")

    _ensure_dir(persist_dir)

    # If persist already complete and not forced, keep it
    sqlite_ok, shards_ok, shard_names = _markers(persist_dir)
    if not force and sqlite_ok and shards_ok and _local_db_usable(persist_dir):
        log.info("[SYNC] Local dir '%s' already complete; skipping download (shards=%s).",
                 persist_dir, shard_names)
    else:
        if not bucket:
            log.info("[SYNC] MINIO_BUCKET_NAME not set; skipping MinIO download.")
        else:
            # Prefijos candidatos: el configurado primero y luego las rutas
            # conocidas. El pipeline de normativas sube a data/chroma_db/;
            # despliegues con MINIO_PREFIX desactualizado encontraban 0 objetos
            # y el bot quedaba sin índice ("No tengo evidencia" para todo).
            candidates = [prefix]
            for alt in ("data/chroma_db/", "chroma_db/"):
                if alt not in candidates:
                    candidates.append(alt)
            try:
                client = _minio_client()
                for pfx in candidates:
                    try:
                        # Confirmar que el prefijo tiene objetos ANTES de tocar
                        # el directorio local (no borrar la DB por un prefijo vacío).
                        objs = list(client.list_objects(bucket, prefix=pfx, recursive=True))
                        if not objs:
                            log.warning("[SYNC] Prefijo '%s' sin objetos; probando siguiente", pfx)
                            continue
                        # Limpiar restos de una DB anterior para no mezclar índices
                        # (shards huérfanos de otro modelo de embeddings).
                        for child in Path(persist_dir).iterdir():
                            try:
                                child.unlink() if child.is_file() else shutil.rmtree(child)
                            except Exception as rm_err:
                                log.warning("[SYNC] No se pudo limpiar %s: %s", child, rm_err)
                        log.info("[SYNC] Descargando minio://%s/%s -> %s (force=%s)",
                                 bucket, pfx, persist_dir, force)
                        count = _download_prefix(client, bucket=bucket, prefix=pfx, local_dir=persist_dir)
                        log.info("[SYNC] Descarga completa. Objetos descargados: %d (prefix=%s)", count, pfx)
                        break
                    except Exception as e:
                        log.warning("[SYNC] Prefijo '%s' falló: %s", pfx, e)
                else:
                    log.error("[SYNC] Ningún prefijo candidato tenía objetos: %s", candidates)
            except Exception as e:
                log.error("[SYNC] Falló la descarga: %s", e)

    # Copy to runtime_dir only when it is a different path from persist_dir.
    # If they are the same, copying would delete the just-downloaded files (rmtree)
    # before trying to copy them, leaving an empty directory.
    same_dir = Path(persist_dir).resolve() == Path(runtime_dir).resolve()
    if same_dir:
        _fix_permissions(persist_dir)
        s_ok, sh_ok, sh_names = _markers(persist_dir)
        log.info("[SYNC] persist_dir == runtime_dir (%s); no copy needed. sqlite=%s, shards=%s, %s",
                 persist_dir, s_ok, sh_ok, sh_names)
    else:
        _ensure_dir(runtime_dir)
        try:
            if Path(runtime_dir).exists():
                shutil.rmtree(runtime_dir)
            shutil.copytree(persist_dir, runtime_dir)
            _fix_permissions(runtime_dir)
            s_ok, sh_ok, sh_names = _markers(runtime_dir)
            log.info("[SYNC] Runtime copy ok → %s  (sqlite=%s, shards=%s, %s)",
                     runtime_dir, s_ok, sh_ok, sh_names)
        except Exception as e:
            log.error("[SYNC] Runtime copy failed: %s", e)
            runtime_dir = persist_dir  # fall back to where the data actually is

    return runtime_dir
