from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

import certifi
import urllib3
from minio import Minio

def _as_bool(value: str | None) -> bool | None:
    """Parsea MINIO_SECURE. None = no definido (caer a la heurística)."""
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _endpoint_and_secure() -> tuple[str, bool]:
    ep = os.getenv("MINIO_PRIVATE_ENDPOINT") or os.getenv("MINIO_PUBLIC_ENDPOINT")
    if not ep:
        raise RuntimeError("MINIO_PRIVATE_ENDPOINT or MINIO_PUBLIC_ENDPOINT must be set.")
    # Este proyecto usa DOS endpoints con seguridad distinta y un solo MINIO_SECURE
    # no puede describir ambos:
    #   - público  "https://...:443"        → TLS terminado en el edge de Railway (secure)
    #   - privado  "bucket.railway.internal:9000" → HTTP plano interno (no secure)
    # Por eso el ESQUEMA explícito de la URL MANDA; MINIO_SECURE solo aplica a
    # endpoints SIN esquema (p. ej. el privado o un TCP Proxy host:puerto), y si
    # tampoco está, se infiere del puerto :443.
    if "://" in ep:
        scheme, rest = ep.split("://", 1)
        return rest.rstrip("/"), scheme == "https"
    secure_override = _as_bool(os.getenv("MINIO_SECURE"))
    if secure_override is not None:
        return ep, secure_override
    return ep, ep.endswith(":443")

def _http_client() -> urllib3.PoolManager:
    """PoolManager con un timeout de CONEXIÓN corto para fallar rápido.

    El cliente Minio por defecto usa connect=read=300s: si el endpoint no es
    alcanzable (p. ej. el endpoint público de Railway apuntando al puerto
    equivocado), el proceso queda colgado 5 minutos antes de fallar. Aquí la
    conexión falla en ~10s mientras el read sigue amplio para no cortar
    transferencias grandes (embeddings, sesión del navegador). Configurable por
    entorno: MINIO_CONNECT_TIMEOUT / MINIO_READ_TIMEOUT / MINIO_HTTP_RETRIES.
    Se replica la verificación TLS que hace Minio internamente (certifi).
    """
    connect = float(os.getenv("MINIO_CONNECT_TIMEOUT", "10"))
    read = float(os.getenv("MINIO_READ_TIMEOUT", "300"))
    retries = int(os.getenv("MINIO_HTTP_RETRIES", "2"))
    return urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=connect, read=read),
        maxsize=10,
        cert_reqs="CERT_REQUIRED",
        ca_certs=os.environ.get("SSL_CERT_FILE") or certifi.where(),
        retries=urllib3.Retry(
            total=retries,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504],
        ),
    )

def _client() -> Minio:
    endpoint, secure = _endpoint_and_secure()
    user = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY")
    pwd  = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY")
    if not user or not pwd:
        raise RuntimeError("MINIO credentials not set (MINIO_ROOT_USER / MINIO_ROOT_PASSWORD).")
    return Minio(
        endpoint, access_key=user, secret_key=pwd, secure=secure,
        http_client=_http_client(),
    )

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

def download_folder(bucket: str, prefix: str, local_dir: str) -> None:
    download_prefix(bucket=bucket, prefix=prefix, dest_dir=local_dir)

def upload_folder(local_dir: str, bucket: str, prefix: str) -> None:
    cli = _client()
    base = Path(local_dir)
    for path in base.rglob("*"):
        if path.is_file():
            key = prefix.rstrip("/") + "/" + path.relative_to(base).as_posix()
            cli.fput_object(bucket, key, str(path))
