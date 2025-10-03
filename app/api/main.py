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
    from app.observability.obs import trace_ctx, span_ctx
except Exception:
    class _DummyCtx:
        def __call__(self, *a, **k): return self
        def __enter__(self): return self
        def __exit__(self, *a): return False
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
    with trace_ctx(name="query_docs", user_id="api", metadata={"agent_key": agent_key, "q": q}) as tr:
        with span_ctx(tr, "agent_answer"):
            answer = _agent.answer(q, agent_key=agent_key, is_admin=admin)
    return {"question": q, "answer": answer, "agent_key": agent_key, "admin": admin}

app.include_router(router)

# --- Telegram webhook (optional) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN:
    from aiogram import Bot, types
    # Reuse your polling dispatcher only if you need; otherwise treat webhook as independent
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
