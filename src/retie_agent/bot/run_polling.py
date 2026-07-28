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

    # 3. Poll hasta que el server esté listo E INICIALIZA la sesión MCP UNA vez.
    #
    #    notebooklm-mcp admite UN solo transporte activo a la vez. El healthcheck
    #    anterior abría el transporte con httpx y DESCARTABA el Mcp-Session-Id, así
    #    que los clientes siguientes (diagnose, keepalive, grafo) no podían
    #    reinicializar ("Already connected to a transport" → HTTP 500) ni tenían un
    #    session id que reusar. Ahora inicializamos con NotebookLMClient, que
    #    PERSISTE el Mcp-Session-Id en disco; todos los demás clientes lo recargan
    #    y comparten ese único transporte.
    from retie_agent.agent.notebooklm_client import NotebookLMClient
    loop = asyncio.get_event_loop()
    deadline = loop.time() + startup_timeout

    def _init_mcp_session() -> bool:
        client = NotebookLMClient(base_url=nlm_url, timeout=10.0)
        # Arranque fresco: descarta cualquier session id viejo (baked en la imagen
        # o de un contenedor anterior) — el server recién arrancado no tiene aún
        # transporte, así que un id viejo provocaría el conflicto que evitamos.
        client._clear_session_cache()
        client._mcp_session = None
        return client.initialize()  # captura y persiste el Mcp-Session-Id

    while loop.time() < deadline:
        await asyncio.sleep(3)
        try:
            ready = await asyncio.get_running_loop().run_in_executor(None, _init_mcp_session)
        except Exception:
            ready = False
        if ready:
            logging.info("[NLM] MCP server ready — sesión MCP inicializada y persistida")
            await _diagnose_nlm(nlm_url)
            return

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

        # OJO: en modo persistente el contexto del navegador se lanza de forma
        # PEREZOSA (en la 1ª ask_question), así que get_health al arranque reporta
        # authenticated=false aunque las cookies existan. Por eso este chequeo es
        # solo informativo — la auth real se confirma en la primera consulta, cuando
        # el server carga browser_state/state.json al lanzar el contexto.
        if client.is_authenticated():
            logging.info("[NLM] Sesión de Google válida ✓")
        else:
            logging.info(
                "[NLM] Auth aún sin verificar al arranque (el navegador se lanza en "
                "la 1ª consulta). Si una consulta falla por auth, re-loguea local: "
                "python setup_notebooklm.py → --pack/--upload."
            )

        configured = getattr(settings, "NOTEBOOKLM_NOTEBOOK_ID", None)
        try:
            resp = client._call_tool("list_notebooks", {})
            content = resp.get("result", {}).get("content", [])
            text = content[0].get("text", "{}") if isinstance(content, list) and content else "{}"
            data = _json.loads(text) if isinstance(text, str) else (text or {})
            # El server envuelve la salida en {"success":true,"data":{...}}; hay que
            # desempaquetar `data` antes de leer `notebooks` (si no, contaba 0).
            if isinstance(data, dict):
                inner = data.get("data", data)
                if isinstance(inner, list):
                    notebooks = inner
                elif isinstance(inner, dict):
                    notebooks = inner.get("notebooks", [])
                else:
                    notebooks = []
            else:
                notebooks = []
            ids = [nb.get("id") for nb in notebooks]
            for nb in notebooks:
                logging.info("[NLM] notebook disponible: [%s] %s",
                             nb.get("id"), nb.get("name") or nb.get("title", "?"))
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


async def _try_recover_nlm_session(nlm_url: str) -> None:
    """Reinicia el servidor MCP descargando la sesión más reciente desde MinIO.

    Se invoca desde dos puntos:
    - keepalive: cuando is_authenticated() devuelve False tras el intervalo normal.
    - on-demand: cuando notebooklm_node señaliza NLM_RECOVERY_NEEDED por un error de auth.

    Si MinIO no tiene sesión válida, el servidor arranca sin auth y los logs
    indican que se requiere login manual (setup_notebooklm.py --upload).
    """
    global _nlm_proc
    logging.info("[NLM] auto-recuperación: reiniciando servidor MCP...")

    if _nlm_proc and _nlm_proc.returncode is None:
        try:
            _nlm_proc.terminate()
            await asyncio.wait_for(_nlm_proc.wait(), timeout=10)
        except (asyncio.TimeoutError, Exception):
            try:
                _nlm_proc.kill()
            except Exception:
                pass
        _nlm_proc = None

    await _setup_nlm_server()

    def _check_auth() -> bool:
        from retie_agent.agent.notebooklm_client import NotebookLMClient
        return NotebookLMClient(base_url=nlm_url, timeout=30.0).is_authenticated()

    try:
        ok = await asyncio.get_running_loop().run_in_executor(None, _check_auth)
        if ok:
            logging.info("[NLM] auto-recuperación exitosa ✓ — sesión de Google restaurada")
        else:
            logging.warning(
                "[NLM] auto-recuperación: servidor reiniciado pero sesión de Google "
                "sigue inválida. Ejecuta setup_notebooklm.py localmente y sube con --upload."
            )
    except Exception as exc:
        logging.warning("[NLM] auto-recuperación: verificación falló: %s", exc)


async def _nlm_keepalive_loop(nlm_url: str) -> None:
    """Mantiene viva la sesión de Google y reacciona a fallos sin intervención manual.

    Cada NOTEBOOKLM_KEEPALIVE_MINUTES:
      1. Toca la sesión vía get_health (Playwright refresca cookies al navegar).
      2. Si sigue autenticada, sube el browser_state renovado a MinIO.
      3. Si la sesión ya venció → intenta auto-recuperación (reinicia el proceso MCP).

    Además revisa cada 60 s el flag NLM_RECOVERY_NEEDED: cuando notebooklm_node
    detecta un error de autenticación lo activa para forzar recuperación inmediata
    sin esperar al próximo intervalo de keepalive.
    """
    from retie_agent.agent.notebooklm_client import NLM_RECOVERY_NEEDED

    minutes = int(getattr(settings, "NOTEBOOKLM_KEEPALIVE_MINUTES", 240))
    if minutes <= 0:
        return

    POLL_INTERVAL = 60
    elapsed = 0

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
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        if NLM_RECOVERY_NEEDED.is_set():
            NLM_RECOVERY_NEEDED.clear()
            elapsed = 0
            logging.info("[NLM] recuperación on-demand solicitada por el grafo")
            await _try_recover_nlm_session(nlm_url)
            continue

        if elapsed < minutes * 60:
            continue

        elapsed = 0
        try:
            ok = await asyncio.get_running_loop().run_in_executor(None, _touch_and_backup)
            if ok:
                logging.info("[NLM] keepalive ✓ — sesión refrescada y respaldada en MinIO")
            else:
                logging.warning("[NLM] keepalive: sesión inválida — iniciando auto-recuperación")
                await _try_recover_nlm_session(nlm_url)
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
