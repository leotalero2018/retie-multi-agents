# app/api/main.py
# FastAPI app: health, query, ingest+debug routers, optional Telegram webhook,
# and simple GridFS file listing/serving (gallery).

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Query, APIRouter, HTTPException
from fastapi.responses import Response, HTMLResponse

from retie_agent.agent.graph import run_graph
from retie_agent.api.debug import router as debug_router
from retie_agent.api import ingest

app = FastAPI(title="RETIE Agent API")

# Routers
app.include_router(debug_router)
app.include_router(ingest.router)

router = APIRouter()

@app.get("/health")
async def health():
    return {"ok": True}

@router.get("/query")
def query_docs(
    q: str = Query(..., description="User question"),
    agent_key: str | None = Query(default=None, description="Route to a specific collection (e.g., plumber|pymupdf)"),
    session_id: str = Query(default="api_query", description="Session ID for chat history"),
):
    """
    Answer a question using the LangGraph pipeline (instrumented for Langfuse).
    If `agent_key` is provided, it switches the collection.
    """
    raw = run_graph(
        q,
        user_id="api",
        session=session_id,
        agent_key=agent_key,
        metadata={"via": "http", "channel": "web"},
    )
    answer = raw.get("formatted_response", raw) if isinstance(raw, dict) else raw
    sources = raw.get("sources", []) if isinstance(raw, dict) else []
    return {
        "question": q,
        "answer": answer,
        "sources": sources,
        "session_id": session_id,
        "agent_key": agent_key,
    }

app.include_router(router)

# --- Telegram webhook (optional) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN:
    from aiogram import Bot, types
    try:
        from retie_agent.bot.run_polling import dp
    except Exception:
        dp = None

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        body = await request.json()
        if dp is None:
            return {"status": "disabled", "reason": "dp not available"}
        update = types.Update.model_validate(body)
        await dp.feed_update(bot, update)
        return {"status": "ok"}
else:
    @app.post("/telegram/webhook")
    async def telegram_webhook_disabled(_: Request):
        return {"status": "disabled", "reason": "TELEGRAM_BOT_TOKEN not configured"}

# --- File listing & serving (GridFS) ---
from retie_agent.services.mongo_store import get_image_bytes, list_recent_files

@app.get("/files")
def files_list(limit: int = 20):
    """
    JSON list of recent saved images (GridFS).
    Each item includes an `id` and a `view_url` you can open in the browser.
    """
    try:
        items = list_recent_files(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"File service error: {e}")
    for it in items:
        it["view_url"] = f"/files/{it['id']}"
    return items

@app.get("/files/{file_id}")
def files_get(file_id: str):
    """
    Stream a single image by id.
    """
    try:
        data, ctype, fname = get_image_bytes(file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"File service error: {e}")
    return Response(
        content=data,
        media_type=ctype,
        headers={"Content-Disposition": f'inline; filename="{fname}"'}
    )

@app.get("/gallery", response_class=HTMLResponse)
def gallery(limit: int = 30):
    """
    Simple HTML gallery to verify saved images quickly.
    """
    items = files_list(limit=limit)
    if isinstance(items, dict) and "detail" in items:  # defensive
        raise HTTPException(status_code=503, detail="File service unavailable")
    blocks = []
    for it in items:  # type: ignore
        caption = (it.get("metadata") or {}).get("caption", "")
        blocks.append(
            f'<div style="margin:12px;display:inline-block;text-align:center">'
            f'<img src="/files/{it["id"]}" '
            f'style="max-width:260px;max-height:260px;display:block;border-radius:8px" />'
            f'<small>{it.get("filename","")}</small><br/>'
            f'<small>{caption}</small>'
            f'</div>'
        )
    html = "<h3>Saved images</h3>" + ("\n".join(blocks) if blocks else "<p>No images yet.</p>")
    return f"<html><body>{html}</body></html>"
