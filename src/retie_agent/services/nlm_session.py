"""Backup and restore the notebooklm-mcp auth state to/from MinIO.

Only the browser_state/ subfolder is backed up — it contains the Playwright
storageState (cookies + localStorage, ~50 KB) that keeps the Google session
alive.  The full chrome_profile/ is OS-specific and too large to backup.

Default data-dir locations (convención `env-paths`; solo Windows lleva `Data`):
  Windows : %LOCALAPPDATA%\\notebooklm-mcp\\Data
  Linux   : ~/.local/share/notebooklm-mcp
  macOS   : ~/Library/Application Support/notebooklm-mcp

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
    """Directorio de datos de notebooklm-mcp según el SO (convención `env-paths`).

    OJO: SOLO Windows añade el sufijo `Data`; macOS y Linux NO. Antes se agregaba
    `/Data` en todos los SO, así que en Railway (Linux) la sesión se restauraba en
    `~/.local/share/notebooklm-mcp/Data/browser_state` mientras el server la busca
    en `~/.local/share/notebooklm-mcp/browser_state` → la sesión de Google nunca se
    cargaba (server con "Notebooks: 0" y sin auth).
    """
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "notebooklm-mcp" / "Data"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "notebooklm-mcp"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "notebooklm-mcp"


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

def _add_session_to_tar(tar: "tarfile.TarFile", browser_state: Path) -> list[str]:
    """Añade al tar la sesión completa: `browser_state/` (cookies de Google) y,
    si existe, `library.json` (registro de notebooks).

    library.json es IMPRESCINDIBLE: sin él, el server arranca con la librería
    vacía y `ask_question` falla con "Notebook not found in library: <id>" aunque
    pases NOTEBOOKLM_NOTEBOOK_ID. Vive como hermano de browser_state, en la raíz
    del data-dir de notebooklm-mcp.
    """
    added: list[str] = []
    tar.add(str(browser_state), arcname="browser_state")
    added.append("browser_state/")
    library = browser_state.parent / "library.json"
    if library.exists():
        tar.add(str(library), arcname="library.json")
        added.append("library.json")
    else:
        logger.warning(
            "[NLM] library.json no encontrado junto a %s — el server tendrá la "
            "librería vacía y ask_question fallará con 'Notebook not found'. "
            "Registra el notebook con: python setup_notebooklm.py", browser_state,
        )
    return added


def upload_nlm_session(source_dir: str | None = None) -> None:
    """Compress the session (browser_state + library.json) and upload it to MinIO.

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
        added = _add_session_to_tar(tar, src)
    size = buf.tell()
    buf.seek(0)

    cli.put_object(bucket, _MINIO_KEY, buf, size, content_type="application/gzip")
    logger.info(
        "[NLM] session uploaded to MinIO (%s/%s, %.1f KB) — incluye: %s",
        bucket, _MINIO_KEY, size / 1024, ", ".join(added),
    )


def pack_nlm_session(source_dir: str | None = None, dest_dir: str | None = None) -> Path:
    """Comprime la sesión (browser_state + library.json) a un .tar.gz LOCAL.

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
        added = _add_session_to_tar(tar, src)

    logger.info(
        "[NLM] session packed → %s (%.1f KB) — incluye: %s",
        out_path, out_path.stat().st_size / 1024, ", ".join(added),
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
    except Exception as exc:
        endpoint = os.getenv("MINIO_PRIVATE_ENDPOINT") or os.getenv("MINIO_PUBLIC_ENDPOINT")
        raise FileNotFoundError(
            f"No NLM session found in MinIO ({bucket}/{_MINIO_KEY}) via {endpoint!r}: {exc}. "
            "Run: python setup_notebooklm.py --upload"
        ) from exc

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
        names = tar.getnames()
        tar.extractall(dest)

    has_library = any(n == "library.json" or n.endswith("/library.json") for n in names)
    logger.info(
        "[NLM] session restored from MinIO → %s (browser_state%s)",
        dest, " + library.json" if has_library else " — SIN library.json (re-empaqueta con --pack)",
    )
