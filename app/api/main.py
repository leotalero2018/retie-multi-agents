# app/api/main.py
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher
from aiogram.types import Update

from app.config import settings
from app.bot.run_polling import router  # reutilizamos handlers
from app.observability.obs import trace_ctx, span_ctx

app = FastAPI()
bot = Bot(token=settings.TELEGRAM_TOKEN)
dp = Dispatcher()
dp.include_router(router)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    body = await request.json()

    # Creamos un trace por request
    with trace_ctx(
        name="telegram_webhook",
        user_id=str(body.get("message", {}).get("from", {}).get("id", "")),
        metadata={"chat_id": body.get("message", {}).get("chat", {}).get("id", "")},
    ) as trace:

        with span_ctx(trace, "feed_update"):
            update = Update.model_validate(body)
            await dp.feed_update(bot, update)

    return {"status": "ok"}
