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
                    timeout=5,
                )
            if resp.status_code < 500:
                logging.info("[NLM] MCP server ready at %s", mcp_url)
                return
        except Exception:
            pass

    logging.warning("[NLM] Server not ready after %ds — NLM disabled for this session", startup_timeout)


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

    # 4️⃣ Start NotebookLM MCP server (if enabled)
    nlm_enabled = str(getattr(settings, "NOTEBOOKLM_ENABLED", "false")).lower() in ("1", "true", "yes")
    if nlm_enabled:
        logging.info("[NLM] NOTEBOOKLM_ENABLED=true — starting MCP server...")
        await _setup_nlm_server()

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
