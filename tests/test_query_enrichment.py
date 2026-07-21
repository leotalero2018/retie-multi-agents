# tests/test_query_enrichment.py
"""Tests de query_enrichment_node (ex condense_node) — Fase 2 del plan v3.

Sin red: create_chat_completion se mockea en el módulo graph. Cubre el
pass-through sin costo, los dos disparadores (historial / complexity=high),
la guarda de sanidad contra reescrituras desproporcionadas y el glosario.
"""
from types import SimpleNamespace

import pytest

from retie_agent.agent import graph as graph_mod
from retie_agent.agent.prompt import (
    CONDENSE_PROMPT,
    QUERY_ENRICHMENT_PROMPT,
)


def _fake_llm(monkeypatch, content):
    """Reemplaza la llamada LLM del nodo; devuelve `content` y cuenta llamadas."""
    calls = {"n": 0}

    def _fake(client, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    monkeypatch.setattr(graph_mod, "create_chat_completion", _fake)
    return calls


def _boom_llm(monkeypatch):
    def _fake(client, **kwargs):
        raise AssertionError("el nodo no debía llamar al LLM en este caso")

    monkeypatch.setattr(graph_mod, "create_chat_completion", _fake)


# ── Pass-through (sin costo) ──────────────────────────────────────────────────

def test_sin_historial_ni_complejidad_es_passthrough(monkeypatch):
    _boom_llm(monkeypatch)
    out = graph_mod._node_query_enrichment({"question": "¿qué es una acometida?"})
    assert out == {"search_query": "¿qué es una acometida?"}


def test_flag_apagado_es_passthrough(monkeypatch):
    _boom_llm(monkeypatch)
    monkeypatch.setattr(graph_mod.settings, "QUERY_REWRITE_ENABLED", "false", raising=False)
    out = graph_mod._node_query_enrichment({
        "question": "¿y eso aplica en BT?",
        "history": [{"role": "user", "content": "requisitos de SPT"}],
    })
    assert out["search_query"] == "¿y eso aplica en BT?"


# ── Disparadores ──────────────────────────────────────────────────────────────

def test_con_historial_reescribe(monkeypatch):
    monkeypatch.setattr(graph_mod.settings, "QUERY_REWRITE_ENABLED", "true", raising=False)
    calls = _fake_llm(
        monkeypatch,
        "¿Los requisitos de sistema de puesta a tierra (SPT) aplican en baja tensión (BT)?",
    )
    out = graph_mod._node_query_enrichment({
        "question": "¿y eso aplica en BT?",
        "history": [{"role": "user", "content": "requisitos de SPT"}],
    })
    assert calls["n"] == 1
    assert "puesta a tierra" in out["search_query"]


def test_complexity_high_dispara_sin_historial(monkeypatch):
    monkeypatch.setattr(graph_mod.settings, "QUERY_REWRITE_ENABLED", "true", raising=False)
    calls = _fake_llm(
        monkeypatch,
        "Requisitos de DPS (dispositivo de protección contra sobretensiones) y de "
        "SPT (sistema de puesta a tierra) para tableros",
    )
    out = graph_mod._node_query_enrichment({
        "question": "requisitos de DPS y SPT para tableros",
        "intent_meta": {"complexity": "high"},
    })
    assert calls["n"] == 1
    assert "sobretensiones" in out["search_query"]


def test_complexity_low_sin_historial_no_dispara(monkeypatch):
    _boom_llm(monkeypatch)
    monkeypatch.setattr(graph_mod.settings, "QUERY_REWRITE_ENABLED", "true", raising=False)
    out = graph_mod._node_query_enrichment({
        "question": "requisitos de tableros",
        "intent_meta": {"complexity": "low"},
    })
    assert out["search_query"] == "requisitos de tableros"


# ── Guardas de sanidad ────────────────────────────────────────────────────────

def test_reescritura_desproporcionada_se_descarta(monkeypatch):
    monkeypatch.setattr(graph_mod.settings, "QUERY_REWRITE_ENABLED", "true", raising=False)
    _fake_llm(monkeypatch, "bla " * 500)  # 2000 chars >> max(300, len(q)*4)
    q = "¿y eso aplica en BT?"
    out = graph_mod._node_query_enrichment({
        "question": q,
        "history": [{"role": "user", "content": "requisitos de SPT"}],
    })
    assert out["search_query"] == q


def test_fallo_del_llm_conserva_la_pregunta(monkeypatch):
    monkeypatch.setattr(graph_mod.settings, "QUERY_REWRITE_ENABLED", "true", raising=False)

    def _fail(client, **kwargs):
        raise RuntimeError("LLM caído")

    monkeypatch.setattr(graph_mod, "create_chat_completion", _fail)
    q = "¿y eso aplica en BT?"
    out = graph_mod._node_query_enrichment({
        "question": q,
        "history": [{"role": "user", "content": "requisitos de SPT"}],
    })
    assert out["search_query"] == q


# ── Prompt y aliases ──────────────────────────────────────────────────────────

def test_prompt_contiene_glosario_y_placeholders():
    assert "SPT = sistema de puesta a tierra" in QUERY_ENRICHMENT_PROMPT
    assert "{history}" in QUERY_ENRICHMENT_PROMPT
    assert "{question}" in QUERY_ENRICHMENT_PROMPT


def test_alias_legacy_apuntan_al_nuevo():
    assert CONDENSE_PROMPT is QUERY_ENRICHMENT_PROMPT
    assert graph_mod._node_condense is graph_mod._node_query_enrichment
