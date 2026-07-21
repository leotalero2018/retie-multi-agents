"""_run_graph_with_feedback edita un único mensaje de estado con el progreso
que emite el grafo (graph.py:_emit_progress) mientras corre en un hilo aparte.

El punto delicado a cubrir: el callback de progreso se invoca desde el hilo
worker donde corre run_graph (nunca desde el loop de asyncio), así que debe
reenviarse con run_coroutine_threadsafe — si esto se rompe, las ediciones
simplemente nunca llegan (sin excepción visible), así que vale la pena una
prueba dedicada en vez de confiar en inspección manual.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from retie_agent.bot import router


class _FakeStatusMessage:
    def __init__(self):
        self.texts = []
        self.deleted = False
        self.edit_text = AsyncMock(side_effect=self._record_edit)
        self.delete = AsyncMock(side_effect=self._record_delete)

    async def _record_edit(self, text, **kwargs):
        self.texts.append(text)

    async def _record_delete(self):
        self.deleted = True


class _FakeMessage:
    def __init__(self):
        self.chat = MagicMock(id=123)
        self.bot = MagicMock()
        self.bot.send_chat_action = AsyncMock()
        self.sent = []
        self.status_message = _FakeStatusMessage()

    async def answer(self, text, **kwargs):
        # Primer mensaje de estado; llamadas posteriores (respuesta final) no
        # se ejercitan aquí — _send_response no se prueba en este archivo.
        self.sent.append(text)
        return self.status_message


def _fake_run_graph_factory(events, delay=0.05):
    """Simula run_graph: emite eventos de progreso desde OTRO hilo (como hace
    el run_in_executor real) y devuelve una respuesta final."""

    def _fake_run_graph(question, *args, progress_callback=None, **kwargs):
        assert kwargs.get("metadata") is not None
        for stage, message in events:
            time.sleep(delay)
            if progress_callback:
                progress_callback({"stage": stage, "message": message, "detail": {}})
        return {"formatted_response": "respuesta final", "sources": []}

    return _fake_run_graph


@pytest.mark.asyncio
async def test_progress_events_are_relayed_from_worker_thread(monkeypatch):
    events = [
        ("search_chroma", "🔎 Buscando en la base normativa…"),
        ("deep_agent_reasoning", "🧠 Pensando…"),
        ("deep_agent_synthesizing", "✍️ Creando la respuesta…"),
        ("enrich_node", "🪶 Mejorando la redacción…"),
    ]
    monkeypatch.setattr(router, "run_graph", _fake_run_graph_factory(events))

    message = _FakeMessage()
    result = await router._run_graph_with_feedback(
        message, "¿qué dice el RETIE?", metadata={"via": "text", "channel": "telegram"},
    )

    assert result == {"formatted_response": "respuesta final", "sources": []}
    # Un solo mensaje de estado creado (no uno por evento) y editado en orden.
    assert message.sent == ["🔎 Buscando en la base normativa…"]
    assert message.status_message.texts == [
        "🧠 Pensando…",
        "✍️ Creando la respuesta…",
        "🪶 Mejorando la redacción…",
    ]
    # Se limpia el mensaje de estado antes de que el caller entregue la respuesta.
    assert message.status_message.deleted is True


@pytest.mark.asyncio
async def test_silence_falls_back_to_rotating_heartbeat(monkeypatch):
    """Una sola llamada bloqueante sin sub-etapas propias (p. ej. NotebookLM o
    la síntesis final del LLM) no debe dejar el chat sin ninguna señal por más
    de _PROGRESS_AFTER_S: el heartbeat genérico rellena el silencio y va
    rotando (no repite el mismo texto en cada tick)."""
    monkeypatch.setattr(router, "_PROGRESS_AFTER_S", 0.05)
    monkeypatch.setattr(router, "_TYPING_REFRESH_S", 0.02)

    def _slow_silent_run_graph(question, *args, progress_callback=None, **kwargs):
        time.sleep(0.3)
        return {"formatted_response": "ok", "sources": []}

    monkeypatch.setattr(router, "run_graph", _slow_silent_run_graph)

    message = _FakeMessage()
    result = await router._run_graph_with_feedback(
        message, "hola", metadata={"via": "text", "channel": "telegram"},
    )

    assert result == {"formatted_response": "ok", "sources": []}
    assert message.sent == [router._HEARTBEAT_MESSAGES[0]]
    # Al menos una rotación real: el segundo heartbeat no repite el primero.
    assert len(message.status_message.texts) >= 1
    assert message.status_message.texts[0] == router._HEARTBEAT_MESSAGES[1]
    assert message.status_message.deleted is True
