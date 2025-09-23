# app/api/main.py
from fastapi import FastAPI, Request
from app.config import settings
from app.observability.obs import trace_ctx, span_ctx

app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}

# Levantar endpoints de Telegram SOLO si hay token
if getattr(settings, "TELEGRAM_TOKEN", None):
    from aiogram import Bot, Dispatcher
    from aiogram.types import Update
    from app.bot.router import router  # <-- CHANGED: import from router.py

    bot = Bot(token=settings.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        body = await request.json()
        with trace_ctx(
            name="telegram_webhook",
            user_id=str(body.get("message", {}).get("from", {}).get("id", "")),
            metadata={"chat_id": body.get("message", {}).get("chat", {}).get("id", "")},
        ) as trace:
            with span_ctx(trace, "feed_update"):
                update = Update.model_validate(body)
                await dp.feed_update(bot, update)
        return {"status": "ok"}
else:
    @app.post("/telegram/webhook")
    async def telegram_webhook_disabled(_: Request):
        return {"status": "disabled", "reason": "TELEGRAM_TOKEN not configured"}
