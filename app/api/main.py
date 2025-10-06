# app/api/main.py
# FastAPI app: health, query, ingest+debug routers, optional Telegram webhook (kept minimal and safe).
from __future__ import annotations

import os
from fastapi import FastAPI, Request, Query, APIRouter

from app.agent.retie_agent import RetieAgent  # new minimal agent
from app.api.debug import router as debug_router
from app.api import ingest


# Optional observability: if not present, silently no-op
try:
    from app.observability.obs import trace_ctx, span_ctx, log_generation
except Exception:
    class _DummyCtx:
        def __call__(self, *a, **k): return self
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def log_generation(*a, **k):  # type: ignore
        return
    trace_ctx = span_ctx = _DummyCtx()

app = FastAPI(title="RETIE Agent API")

# Include routers
app.include_router(debug_router)
app.include_router(ingest.router)

# Local routes
router = APIRouter()
_agent = RetieAgent()

@app.get("/health")
async def health():
    return {"ok": True}

@router.get("/query")
def query_docs(
    q: str = Query(..., description="User question"),
    agent_key: str | None = Query(default=None, description="Route to a specific collection (e.g., plumber|pymupdf)"),
    admin: bool = Query(default=False, description="If true, include sources in the answer"),
):
    """
    Answer a question using the embedded Chroma DB via RetieAgent.
    If `agent_key` is provided, it switches the collection (plumber/pymupdf).
    """
    with trace_ctx(name="api_query", user_id="api", metadata={"agent_key": agent_key, "q": q, "admin": admin}) as tr:
        with span_ctx(tr, "agent_answer"):
            answer = _agent.answer(q, agent_key=agent_key, is_admin=admin)
        # record the final answer that the API returns
        try:
            log_generation(tr, "final_answer", q, answer, model="", usage=None, metadata={"agent_key": agent_key, "admin": admin})
        except Exception:
            pass
    return {"question": q, "answer": answer, "agent_key": agent_key, "admin": admin}

app.include_router(router)

# --- Telegram webhook (optional) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN:
    from aiogram import Bot, types
    try:
        from app.bot.run_polling import dp
    except Exception:
        dp = None

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        body = await request.json()
        with trace_ctx(name="telegram_webhook", user_id=str(body.get("message", {}).get("from", {}).get("id", ""))):
            if dp is None:
                return {"status": "disabled", "reason": "dp not available"}
            update = types.Update.model_validate(body)
            await dp.feed_update(bot, update)
        return {"status": "ok"}
else:
    @app.post("/telegram/webhook")
    async def telegram_webhook_disabled(_: Request):
        return {"status": "disabled", "reason": "TELEGRAM_BOT_TOKEN not configured"}
    
# --- File serving (GridFS) ---
from fastapi import HTTPException
from fastapi.responses import Response
try:
    from app.services.mongo_store import get_image_bytes
except Exception:
    get_image_bytes = None  # degrade gracefully if Mongo not configured

@app.get("/files/{file_id}")
def get_file(file_id: str):
    if get_image_bytes is None:
        raise HTTPException(status_code=503, detail="File service disabled")
    try:
        data, ctype, fname = get_image_bytes(file_id)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")
    return Response(
        content=data,
        media_type=ctype,
        headers={"Content-Disposition": f'inline; filename="{fname}"'}
    )
   
