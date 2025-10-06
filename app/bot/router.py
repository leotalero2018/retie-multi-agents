# ...top of file stays the same...
from app.observability.obs import trace_ctx, span_ctx, log_generation
from app.config import settings

# ...

@router.message(F.text)
async def on_text(message: Message):
    q = (message.text or "").strip()
    agent_key = CHAT_AGENT.get(message.chat.id, DEFAULT_AGENT)

    LAST_QUERY[message.chat.id] = q
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    user_id = str(message.from_user.id) if message.from_user else None
    meta = {"agent_key": agent_key, "chat_id": message.chat.id}

    # ---- tracing envelope (root span) ----
    with trace_ctx("telegram.message", user_id=user_id, metadata=meta) as tr:
        # child span for the agent pipeline
        with span_ctx(tr, "agent.answer", metadata={"question": q}):
            loop = asyncio.get_running_loop()
            raw_resp = await loop.run_in_executor(
                None,
                lambda: _agent.answer(
                    q,
                    agent_key=agent_key,
                    is_admin=_is_admin(message.from_user.id if message.from_user else None),
                ),
            )

        # optional: record the LLM step explicitly as a Langfuse Generation
        try:
            # If your agent already made multiple LLM calls you can move this into the agent itself.
            log_generation(
                tr,
                name="openai.chat",
                input_text=q,
                output_text=raw_resp,
                model=getattr(settings, "CHAT_MODEL", "gpt-4o-mini"),
                usage={},  # fill if you track token usage
                metadata={"agent_key": agent_key},
            )
        except Exception:
            pass

    is_admin = _is_admin(message.from_user.id if message.from_user else None)
    resp = _clean_for_user(raw_resp, is_admin)
    await message.answer(resp)
