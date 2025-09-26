# app/api/main.py
from fastapi import FastAPI, Request, Query, APIRouter
from app.config import settings
from app.observability.obs import trace_ctx, span_ctx
from app.api.debug import router as debug_router
from app.api import ingest
from app.llm.provider import chat_answer  # usa el que ya tienes en provider.py
from app.llm.chroma_stub import get_vector_store  # stub temporal si no tienes db

# Crear la app
app = FastAPI()

# Routers existentes
app.include_router(debug_router)
app.include_router(ingest.router)

# Router local para endpoints propios
router = APIRouter()


@app.get("/health")
async def health():
    return {"ok": True}


@router.get("/query")
def query_docs(q: str = Query(..., description="Pregunta del usuario")):
    # 1. Recuperar documentos (stub o Chroma si luego lo conectas)
    vs = get_vector_store()
    results = vs.similarity_search(q, k=3)

    # 2. Armar contexto
    context = "\n".join([r.page_content for r in results])

    # 3. Llamar al modelo (usa el provider que ya tienes funcionando)
    answer = chat_answer(
        prompt=f"Pregunta: {q}\nContexto:\n{context}",
        blocks=[],
        question=q,
        is_admin=False
    )

    return {"question": q, "answer": answer, "context": context}


# Incluir router local
app.include_router(router)


# --- TELEGRAM ---
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
