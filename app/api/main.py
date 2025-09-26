# app/api/main.py
from fastapi import FastAPI, Request
from app.config import settings
from app.observability.obs import trace_ctx, span_ctx
from app.api.debug import router as debug_router


app = FastAPI()

app.include_router(debug_router)
 
@app.get("/health")
async def health():
    return {"ok": True}

# Levantar endpoints de Telegram SOLO si hay token
if getattr(settings, "TELEGRAM_TOKEN", None):
    from aiogram import Bot, types
    from app.bot.run_polling import dp

    bot = Bot(token=settings.TELEGRAM_TOKEN)

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        body = await request.json()
        with trace_ctx(
            name="telegram_webhook",
            user_id=str(body.get("message", {}).get("from", {}).get("id", "")),
            metadata={"chat_id": body.get("message", {}).get("chat", {}).get("id", "")},
        ) as trace:
            with span_ctx(trace, "feed_update"):
                update = types.Update.model_validate(body)
                await dp.feed_update(bot, update)
        return {"status": "ok"}
else:
    @app.post("/telegram/webhook")
    async def telegram_webhook_disabled(_: Request):
        return {"status": "disabled", "reason": "TELEGRAM_TOKEN not configured"}
