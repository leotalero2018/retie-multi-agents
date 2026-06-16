"""Backup and restore the notebooklm-mcp auth state to/from MinIO.

Only the browser_state/ subfolder is backed up — it contains the Playwright
storageState (cookies + localStorage, ~50 KB) that keeps the Google session
alive.  The full chrome_profile/ is OS-specific and too large to backup.

Default data-dir locations (notebooklm-mcp convention):
  Windows : %LOCALAPPDATA%\\notebooklm-mcp\\Data
  Linux   : ~/.local/share/notebooklm-mcp/Data
  macOS   : ~/Library/Application Support/notebooklm-mcp/Data

MinIO key:  <bucket>/nlm-browser-state.tar.gz
"""
from __future__ import annotations

import io
import logging
import os
import platform
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

_MINIO_KEY = "nlm-browser-state.tar.gz"


# ── OS-default data directory ─────────────────────────────────────────────────

def _default_data_dir() -> Path:
    """Returns the OS-default directory where notebooklm-mcp stores its data."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif system == "Darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "notebooklm-mcp" / "Data"


def default_browser_state_dir() -> Path:
    """Returns the full path to the browser_state/ subfolder."""
    return _default_data_dir() / "browser_state"


# ── MinIO helpers ─────────────────────────────────────────────────────────────

def _minio_client():
    from retie_agent.services.storage_minio import _client
    return _client()


def _bucket() -> str:
    from retie_agent.config import settings
    bucket = (
        getattr(settings, "NOTEBOOKLM_SESSION_BUCKET", "")
        or os.getenv("S3_BUCKET_NAME", "")
        or os.getenv("MINIO_BUCKET_NAME", "")
        or getattr(settings, "S3_BUCKET_NAME", "")
    )
    if not bucket:
        raise RuntimeError(
            "No MinIO bucket configured. Set S3_BUCKET_NAME, MINIO_BUCKET_NAME, or NOTEBOOKLM_SESSION_BUCKET."
        )
    return bucket


# ── Public API ────────────────────────────────────────────────────────────────

def upload_nlm_session(source_dir: str | None = None) -> None:
    """Compress the browser_state directory and upload it to MinIO.

    source_dir: path to the browser_state/ folder.  Defaults to the OS-specific
                notebooklm-mcp default location.
    """
    src = Path(source_dir) if source_dir else default_browser_state_dir()
    if not src.exists():
        raise FileNotFoundError(
            f"browser_state directory not found: {src}\n"
            "Make sure you authenticated first with: python setup_notebooklm.py"
        )

    cli = _minio_client()
    bucket = _bucket()
    if not cli.bucket_exists(bucket):
        cli.make_bucket(bucket)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src), arcname="browser_state")
    size = buf.tell()
    buf.seek(0)

    cli.put_object(bucket, _MINIO_KEY, buf, size, content_type="application/gzip")
    logger.info(
        "[NLM] browser_state uploaded to MinIO (%s/%s, %.1f KB)",
        bucket, _MINIO_KEY, size / 1024,
    )


def pack_nlm_session(source_dir: str | None = None, dest_dir: str | None = None) -> Path:
    """Comprime browser_state/ a un .tar.gz LOCAL, sin tocar MinIO.

    Pensado para cuando MinIO no es alcanzable: genera el archivo para subirlo
    manualmente al bucket. El nombre del archivo coincide EXACTAMENTE con la clave
    que Railway descarga (`_MINIO_KEY`), para que la subida manual use esa clave.

    source_dir: carpeta browser_state/ (por defecto, la ruta del SO).
    dest_dir:   carpeta de salida (por defecto ./sessions).
    Devuelve la ruta del .tar.gz generado.
    """
    src = Path(source_dir) if source_dir else default_browser_state_dir()
    if not src.exists():
        raise FileNotFoundError(
            f"browser_state directory not found: {src}\n"
            "Make sure you authenticated first with: python setup_notebooklm.py"
        )

    out_dir = Path(dest_dir) if dest_dir else (Path.cwd() / "sessions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _MINIO_KEY

    with tarfile.open(str(out_path), mode="w:gz") as tar:
        tar.add(str(src), arcname="browser_state")

    logger.info(
        "[NLM] browser_state packed → %s (%.1f KB)",
        out_path, out_path.stat().st_size / 1024,
    )
    return out_path


def session_target() -> tuple[str, str]:
    """(bucket, key) donde Railway espera la sesión — para la subida manual."""
    return _bucket(), _MINIO_KEY


def download_nlm_session(dest_dir: str | None = None) -> None:
    """Download and restore the browser_state archive from MinIO.

    dest_dir: parent directory where browser_state/ will be extracted.
              Defaults to the OS-specific notebooklm-mcp default location
              (i.e. the Data/ folder).
    """
    cli = _minio_client()
    bucket = _bucket()

    try:
        cli.stat_object(bucket, _MINIO_KEY)
    except Exception:
        raise FileNotFoundError(
            f"No NLM session found in MinIO ({bucket}/{_MINIO_KEY}). "
            "Run: python setup_notebooklm.py --upload"
        )

    buf = io.BytesIO()
    response = cli.get_object(bucket, _MINIO_KEY)
    try:
        for chunk in response.stream(32 * 1024):
            buf.write(chunk)
    finally:
        response.close()
        response.release_conn()

    buf.seek(0)
    dest = Path(dest_dir) if dest_dir else _default_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(dest)

    restored = dest / "browser_state"
    logger.info("[NLM] browser_state restored from MinIO → %s", restored)
