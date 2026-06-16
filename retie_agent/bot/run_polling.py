# app/bot/run_polling.py
from __future__ import annotations
import asyncio
import logging
import os
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from retie_agent.config import settings
from pipeline_indexacion.bootstrap_sync import sync_chroma_from_minio
from retie_agent.bot.router import router

logging.basicConfig(level=logging.INFO, force=True)

# ── NotebookLM MCP server lifecycle ──────────────────────────────────────────

_nlm_proc: "asyncio.subprocess.Process | None" = None


async def _drain_nlm_logs(proc: "asyncio.subprocess.Process") -> None:
    """Reemite el stdout/stderr del servidor MCP en los logs del agente.

    Imprescindible: sin esto el output del MCP cae en un PIPE que nadie lee, así
    que (1) la causa real de los "HTTP 500 internal server error" (errores de
    Playwright/navegador) queda OCULTA, y (2) el buffer del PIPE se llena (~64 KB)
    y BLOQUEA el proceso MCP cuando intenta escribir más → cuelgues y 500s.
    """
    stream = proc.stdout
    if stream is None:
        return
    try:
        while True:
            line = await stream.readline()
            if not line:  # EOF: el proceso MCP terminó
                break
            logging.info("[NLM-MCP] %s", line.decode(errors="replace").rstrip())
    except Exception as exc:
        logging.warning("[NLM] log drain terminó: %s", exc)


async def _setup_nlm_server() -> None:
    """Download the Playwright session from MinIO, start notebooklm-mcp in the
    background, and wait until the server is ready to accept requests.

    Failures are non-fatal: the bot continues in Chroma-only mode.
    """
    global _nlm_proc

    session_dir = getattr(settings, "NOTEBOOKLM_SESSION_DIR", "/tmp/nlm-session")
    startup_timeout = int(getattr(settings, "NOTEBOOKLM_STARTUP_TIMEOUT", 60))
    nlm_url = getattr(settings, "NOTEBOOKLM_URL", "http://localhost:3000")
    try:
        port = int(nlm_url.rsplit(":", 1)[-1].rstrip("/"))
    except ValueError:
        port = 3000

    # 1. Restore Playwright browser_state from MinIO to the OS-default location
    #    so notebooklm-mcp picks it up automatically (no env var needed).
    try:
        from retie_agent.services.nlm_session import download_nlm_session
        download_nlm_session()  # restores to OS default: ~/.local/share/notebooklm-mcp/Data/
    except FileNotFoundError as exc:
        logging.warning("[NLM] %s — starting without cached session (Google login required)", exc)
    except Exception as exc:
        logging.warning("[NLM] MinIO session download failed: %s — continuing anyway", exc)

    # 2. Launch the MCP server process
    env = {
        **os.environ,
        "PLAYWRIGHT_HEADLESS": "1",
    }
    cmd = [
        "npx", "--yes", "notebooklm-mcp@latest",
        "--transport", "http",
        "--port", str(port),
    ]

    try:
        _nlm_proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        logging.info("[NLM] MCP server started (pid=%d, port=%d)", _nlm_proc.pid, port)
        # Drena los logs del MCP a los nuestros: revela la causa real de los 500
        # y evita que el PIPE sin leer se llene y cuelgue el proceso.
        asyncio.create_task(_drain_nlm_logs(_nlm_proc))
    except FileNotFoundError:
        logging.warning("[NLM] 'npx' not found — install Node.js 20+ or set NOTEBOOKLM_ENABLED=false")
        return
    except Exception as exc:
        logging.warning("[NLM] Failed to start MCP server: %s", exc)
        return

    # 3. Poll until the server is ready (non-blocking)
    import httpx
    mcp_url = f"http://localhost:{port}/mcp"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + startup_timeout

    while loop.time() < deadline:
        await asyncio.sleep(3)
        try:
            async with httpx.AsyncClient() as hc:
                resp = await hc.post(
                    mcp_url,
                    json={
                        "jsonrpc": "2.0", "id": 0,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "healthcheck", "version": "0"},
                        },
                    },
                    headers={
                        "Content-Type": "application/json",
                        # Sin Accept dual el server responde 406 Not Acceptable.
                        "Accept": "application/json, text/event-stream",
                    },
                    timeout=5,
                )
            if resp.status_code < 500:
                logging.info("[NLM] MCP server ready at %s", mcp_url)
                await _diagnose_nlm(nlm_url)
                return
        except Exception:
            pass

    logging.warning("[NLM] Server not ready after %ds — NLM disabled for this session", startup_timeout)


async def _diagnose_nlm(nlm_url: str) -> None:
    """Diagnóstico de arranque: ¿hay sesión de Google? ¿qué notebooks existen?

    Convierte el genérico "HTTP 500 internal server error" en logs accionables:
    sin esto era imposible saber si el problema era la sesión vencida o un
    NOTEBOOKLM_NOTEBOOK_ID que no existe en la librería del servidor.
    """
    import json as _json

    def _run_diagnosis() -> None:
        from retie_agent.agent.notebooklm_client import NotebookLMClient
        client = NotebookLMClient(base_url=nlm_url, timeout=30.0)

        if client.is_authenticated():
            logging.info("[NLM] Sesión de Google válida ✓")
        else:
            logging.warning(
                "[NLM] ⚠️ SIN sesión de Google válida — las consultas a NotebookLM "
                "fallarán. Regenera la sesión en local (python setup_notebooklm.py) "
                "y súbela a MinIO con: python setup_notebooklm.py --upload"
            )

        configured = getattr(settings, "NOTEBOOKLM_NOTEBOOK_ID", None)
        try:
            resp = client._call_tool("list_notebooks", {})
            content = resp.get("result", {}).get("content", [])
            text = content[0].get("text", "{}") if isinstance(content, list) and content else "{}"
            data = _json.loads(text) if isinstance(text, str) else (text or {})
            notebooks = data.get("notebooks", []) if isinstance(data, dict) else []
            ids = [nb.get("id") for nb in notebooks]
            for nb in notebooks:
                logging.info("[NLM] notebook disponible: [%s] %s", nb.get("id"), nb.get("title", "?"))
            if configured and ids and configured not in ids:
                logging.warning(
                    "[NLM] ⚠️ NOTEBOOKLM_NOTEBOOK_ID=%r NO está en la librería del "
                    "servidor (disponibles: %s). Corrige la variable en Railway.",
                    configured, ids,
                )
            elif not notebooks:
                logging.warning(
                    "[NLM] La librería del servidor no tiene notebooks registrados; "
                    "agrégalo con setup_notebooklm.py y vuelve a subir la sesión."
                )
        except Exception as exc:
            logging.warning("[NLM] No se pudo listar notebooks: %s", exc)

    try:
        await asyncio.get_running_loop().run_in_executor(None, _run_diagnosis)
    except Exception as exc:
        logging.warning("[NLM] Diagnóstico falló: %s", exc)


async def _nlm_keepalive_loop(nlm_url: str) -> None:
    """Mantiene viva la sesión de Google sin intervención manual.

    Cada NOTEBOOKLM_KEEPALIVE_MINUTES:
      1. Toca la sesión vía get_health (Playwright refresca cookies al navegar).
      2. Si sigue autenticada, sube el browser_state RENOVADO a MinIO — así los
         próximos deploys arrancan con cookies frescas en vez del snapshot
         original que envejece hasta vencer.
    Una sesión ya vencida NO se puede resucitar desde código (el login de
    Google requiere interacción humana): este loop evita llegar a ese punto.
    """
    minutes = int(getattr(settings, "NOTEBOOKLM_KEEPALIVE_MINUTES", 240))
    if minutes <= 0:
        return
    interval = minutes * 60

    def _touch_and_backup() -> bool:
        from retie_agent.agent.notebooklm_client import NotebookLMClient
        from retie_agent.services.nlm_session import upload_nlm_session, default_browser_state_dir

        client = NotebookLMClient(base_url=nlm_url, timeout=60.0)
        if not client.is_authenticated():
            return False
        src = default_browser_state_dir()
        if src.exists():
            upload_nlm_session(str(src))
        return True

    while True:
        await asyncio.sleep(interval)
        try:
            ok = await asyncio.get_running_loop().run_in_executor(None, _touch_and_backup)
            if ok:
                logging.info("[NLM] keepalive ✓ — sesión refrescada y respaldada en MinIO")
            else:
                logging.warning(
                    "[NLM] keepalive: la sesión de Google ya NO es válida — se "
                    "requiere un login manual (setup_notebooklm.py) y --upload."
                )
        except Exception as exc:
            logging.warning("[NLM] keepalive falló: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────

dp = Dispatcher()
dp.include_router(router)


def _clear_if_placeholder(path: str) -> None:
    p = Path(path)
    sqlite = p / "chroma.sqlite3"
    has_shard = any(p.glob("*.db"))
    if p.exists() and p.is_dir() and (not sqlite.exists()) and (not has_shard):
        for child in p.iterdir():
            try:
                if child.is_file():
                    child.unlink()
                else:
                    import shutil
                    shutil.rmtree(child)
            except Exception as e:
                logging.warning("[BOOT] Could not remove %s: %s", child, e)
        logging.info("[BOOT] Cleared placeholder contents in %s", path)


async def main() -> None:
    # 1️⃣ Sync from MinIO → local runtime dir
    try:
        runtime_dir = sync_chroma_from_minio() or settings.CHROMA_DB_DIR
    except Exception as e:
        logging.warning("[SYNC] MinIO sync failed: %s. Using local directory.", e)
        runtime_dir = settings.CHROMA_DB_DIR

    os.environ["CHROMA_DB_DIR"] = runtime_dir
    os.environ["CHROMA_PERSIST_DIR"] = runtime_dir
    settings.CHROMA_DB_DIR = runtime_dir

    from retie_agent.retriever import chroma_client as cc
    cc.set_persist_dir(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    _clear_if_placeholder(runtime_dir)

    logging.info("[BOOT] Using COLLECTION_NAME=%s | CHROMA_DIR=%s",
                 settings.COLLECTION_NAME, settings.CHROMA_DB_DIR)

    # 2️⃣ Quick check
    try:
        files = glob.glob(os.path.join(settings.CHROMA_DB_DIR, "*"))
        logging.info("[LS] Found %d files under %s", len(files), settings.CHROMA_DB_DIR)
    except Exception as e:
        logging.error("[LS] ERROR listing files: %s", e)

    # 3️⃣ Optional: list Chroma collections
    try:
        import chromadb
        cli = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        cols = cli.list_collections()
        for c in cols:
            try:
                cnt = cli.get_collection(c.name).count()
            except Exception as e:
                cnt = f"error: {e}"
            logging.info("   - %s: %s items", c.name, cnt)
    except Exception as e:
        logging.warning("[CHK2] Skipped Chroma check: %s", e)

    # 4️⃣ Start NotebookLM MCP server (if enabled) + keepalive de sesión
    nlm_enabled = str(getattr(settings, "NOTEBOOKLM_ENABLED", "false")).lower() in ("1", "true", "yes")
    if nlm_enabled:
        logging.info("[NLM] NOTEBOOKLM_ENABLED=true — starting MCP server...")
        await _setup_nlm_server()
        nlm_url = getattr(settings, "NOTEBOOKLM_URL", "http://localhost:3000")
        asyncio.create_task(_nlm_keepalive_loop(nlm_url))

    # 5️⃣ Start Telegram bot
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not token:
        logging.error("❌ Missing TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN). Cannot start bot.")
        while True:
            await asyncio.sleep(60)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    logging.info("🤖 Starting RETIE bot polling loop...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
