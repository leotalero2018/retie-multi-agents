"""E8 — Publicación atómica versionada en MinIO (con reintentos y reanudación).

1. Sube out_dir completo (Chroma + assets + registro + manifest + reporte)
   a un prefijo VERSIONADO:  {versions_prefix}/v{timestamp}/...
   - Cada archivo se reintenta con backoff exponencial y reconexión.
   - Los archivos ya presentes en el prefijo (mismo tamaño) se SALTAN →
     una publicación interrumpida se reanuda con `publish --version vXXXX`.
2. Solo al completar, actualiza el puntero {versions_prefix}/latest.json.
3. Espeja al prefijo legacy (default data/chroma_db/) en modo SYNC seguro:
   primero sube/sobrescribe lo nuevo, al final poda lo obsoleto (nunca
   "borrar todo y subir", que dejaba al bot sin índice si fallaba a mitad).

Gate: si index_report.json tiene ok=false, aborta salvo --force.
Rollback: reapuntar latest.json a una versión anterior (no se borran).
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import V2Config

log = logging.getLogger(__name__)

_UPLOAD_ATTEMPTS = 6


def _minio_client():
    from minio import Minio
    endpoint = (os.getenv("MINIO_ENDPOINT")
                or os.getenv("MINIO_PUBLIC_ENDPOINT")
                or os.getenv("MINIO_PRIVATE_ENDPOINT"))
    if not endpoint:
        raise RuntimeError("Falta MINIO_ENDPOINT / MINIO_PUBLIC_ENDPOINT en el entorno")
    access = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("Faltan credenciales MinIO (MINIO_ROOT_USER/MINIO_ROOT_PASSWORD)")
    if "://" in endpoint:
        scheme, endpoint = endpoint.split("://", 1)
        secure = scheme == "https"
    else:
        secure = endpoint.endswith(":443")
    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)


class _ClientPool:
    """Cliente MinIO reciclable: ante un error de conexión se recrea (pool
    HTTP fresco) en el siguiente intento."""

    def __init__(self):
        self._client = None

    def get(self, refresh: bool = False):
        if refresh or self._client is None:
            self._client = _minio_client()
        return self._client


def _remote_sizes(client, bucket: str, prefix: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        for obj in client.list_objects(bucket, prefix=prefix.rstrip("/") + "/", recursive=True):
            out[obj.object_name] = obj.size or 0
    except Exception as exc:
        log.warning("[E8] no se pudo listar '%s' (se sube todo): %s", prefix, exc)
    return out


def _upload_file(pool: _ClientPool, bucket: str, key: str, path: Path) -> None:
    last: Exception | None = None
    for attempt in range(_UPLOAD_ATTEMPTS):
        try:
            pool.get(refresh=attempt > 0).fput_object(bucket, key, str(path))
            return
        except Exception as exc:
            last = exc
            wait = min(60, 3 * (2 ** attempt))
            log.warning("[E8] %s: intento %d/%d falló (%s) — reintento en %ds",
                        key, attempt + 1, _UPLOAD_ATTEMPTS, str(exc)[:120], wait)
            time.sleep(wait)
    raise RuntimeError(f"No se pudo subir {key} tras {_UPLOAD_ATTEMPTS} intentos") from last


def _upload_tree(pool: _ClientPool, bucket: str, prefix: str, local_dir: Path,
                 *, skip_existing: bool = True) -> Dict[str, int]:
    """Sube local_dir a bucket/prefix. Devuelve {"uploaded": n, "skipped": n}.
    Con skip_existing, los objetos remotos con el mismo tamaño se omiten
    (reanudación de publicaciones interrumpidas)."""
    files = [p for p in sorted(local_dir.rglob("*")) if p.is_file()]
    remote = _remote_sizes(pool.get(), bucket, prefix) if skip_existing else {}

    uploaded = skipped = 0
    total = len(files)
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}"
        if skip_existing and remote.get(key) == path.stat().st_size:
            skipped += 1
            continue
        _upload_file(pool, bucket, key, path)
        uploaded += 1
        done = uploaded + skipped
        if uploaded % 25 == 0:
            log.info("[E8]   … %d/%d archivos (%d subidos, %d ya estaban)",
                     done, total, uploaded, skipped)
    log.info("[E8] %s/: %d subidos, %d ya estaban (total %d)",
             prefix, uploaded, skipped, total)
    return {"uploaded": uploaded, "skipped": skipped}


def _prune_extraneous(pool: _ClientPool, bucket: str, prefix: str, local_dir: Path) -> int:
    """Elimina del prefijo remoto los objetos que ya no existen localmente
    (se ejecuta DESPUÉS de subir lo nuevo — sync seguro, nunca delete-first)."""
    from minio.deleteobjects import DeleteObject
    expected = {
        f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}"
        for p in local_dir.rglob("*") if p.is_file()
    }
    remote = _remote_sizes(pool.get(), bucket, prefix)
    stale = [k for k in remote if k not in expected]
    if not stale:
        return 0
    errors = list(pool.get().remove_objects(bucket, [DeleteObject(k) for k in stale]))
    for e in errors:
        log.warning("[E8] error podando: %s", e)
    return len(stale) - len(errors)


def publish(cfg: V2Config, *, force: bool = False, mirror_legacy: bool = True,
            version: Optional[str] = None) -> str:
    """Publica out_dir. `version` reanuda una publicación interrumpida
    (p. ej. --version v20260612-201021). Devuelve el prefijo publicado."""
    report_path = cfg.out_dir / "index_report.json"
    if not report_path.exists():
        raise RuntimeError("No existe index_report.json — ejecuta 'build' o 'validate' antes de publicar")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ok") and not force:
        raise RuntimeError(
            "El reporte tiene gates fallidos:\n  - "
            + "\n  - ".join(report.get("gates", []))
            + "\nResuélvelos o publica con --force."
        )
    if not report.get("ok") and force:
        log.warning("[E8] Publicando CON gates fallidos (--force)")

    pool = _ClientPool()
    bucket = cfg.bucket
    if not pool.get().bucket_exists(bucket):
        log.warning("[E8] bucket '%s' no existe — se crea", bucket)
        pool.get().make_bucket(bucket)

    if version:
        if not version.startswith("v"):
            version = "v" + version
        log.info("[E8] Reanudando publicación de la versión %s", version)
    else:
        version = time.strftime("v%Y%m%d-%H%M%S")
    vprefix = f"{cfg.versions_prefix}/{version}"

    log.info("[E8] Subiendo índice a minio://%s/%s/ ...", bucket, vprefix)
    stats = _upload_tree(pool, bucket, vprefix, cfg.out_dir, skip_existing=True)

    # Puntero latest (atómico: se escribe DESPUÉS de completar la subida)
    pointer = json.dumps({
        "version": version,
        "prefix": vprefix,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunks": report.get("chunks"),
        "tables_total": report.get("tables_total"),
    }).encode()
    pool.get().put_object(bucket, f"{cfg.versions_prefix}/latest.json",
                          io.BytesIO(pointer), len(pointer),
                          content_type="application/json")
    log.info("[E8] latest.json → %s", version)

    # Espejo legacy en modo sync seguro (upload primero, poda al final)
    if mirror_legacy:
        legacy = cfg.legacy_prefix
        log.info("[E8] Espejando al prefijo legacy '%s/' (lo que consume el bot)...", legacy)
        _upload_tree(pool, bucket, legacy, cfg.out_dir, skip_existing=True)
        pruned = _prune_extraneous(pool, bucket, legacy, cfg.out_dir)
        log.info("[E8]   espejo completo (%d objetos obsoletos podados)", pruned)

    log.info("[E8] ✅ Publicación completa: %s (rollback = reapuntar latest.json)", vprefix)
    return vprefix


def list_versions(cfg: V2Config) -> List[str]:
    client = _minio_client()
    seen = set()
    for obj in client.list_objects(cfg.bucket, prefix=cfg.versions_prefix + "/", recursive=True):
        parts = obj.object_name.split("/")
        if len(parts) >= 2 and parts[1].startswith("v"):
            seen.add(parts[1])
    return sorted(seen)
