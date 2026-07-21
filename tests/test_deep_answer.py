# tests/test_deep_answer.py
"""Tests del deep agent en answer_node (Fase 4 del plan v3) — sin red.

Estilo test_gemini_spike.py: el loop de deepagents se mockea vía
deepagents.create_deep_agent (atributo del módulo, resuelto en call-time por
run_deep_answer), el retriever vía retie_agent.retriever.retrieve.search y el
cliente Gemini vía deep_answer._get_gemini. Cubre: registry de skills, tools
que acumulan evidencia y degradan sin lanzar, construcción dinámica, mensaje
inicial con directivas, fallbacks de run_deep_answer y el contrato intocable
de _node_answer.
"""
import time

import pytest

from retie_agent.agent import deep_answer as da
from retie_agent.agent.deep_answer import (
    NO_EVIDENCE_PHRASE,
    DeepAnswerResult,
    EvidenceLog,
    _build_initial_message,
    _build_tools,
    _chat_model_for,
    run_deep_answer,
)
from retie_agent.agent.skills import DEFAULT_SKILL, resolve_skill


def _hits(source="RETIE.pdf", page=10, text="contenido del chunk"):
    return [{"text": text, "score": 0.2, "meta": {"source": source, "page": page}}]


class _FakeGemini:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available

    def ask_question(self, query):
        return "respuesta gemini", [{"title": "fuente"}]


class _FakeAgent:
    def __init__(self, result=None, delay=0.0, exc=None):
        self._result = result
        self._delay = delay
        self._exc = exc
        self.last_config = None

    def invoke(self, payload, config=None):
        self.last_config = config
        if self._delay:
            time.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._result


def _ai(content, usage=None):
    from langchain_core.messages import AIMessage

    return AIMessage(content=content, usage_metadata=usage)


# ── Registry de skills ────────────────────────────────────────────────────────

def test_skill_tabla_con_post_node():
    spec = resolve_skill("tabla", "markdown")
    assert spec.name == "table"
    assert spec.post_node == "table_node"
    assert spec.max_tokens == 4000


def test_skill_por_formato_tabla_con_intent_puntual():
    assert resolve_skill("puntual", "markdown_table").name == "table"


def test_skill_verificacion_y_procedimiento():
    assert resolve_skill("verificacion", "markdown").name == "verification"
    assert resolve_skill("procedimiento", None).name == "steps"


def test_skill_desconocida_cae_al_default():
    assert resolve_skill("desconocido", "markdown") is DEFAULT_SKILL
    assert resolve_skill(None, None) is DEFAULT_SKILL


def test_skills_cargadas_desde_md():
    """Las skills viven como .md en retie_agent/agent/skills/ (frontmatter + directivas)."""
    from retie_agent.agent import skills as skills_pkg

    names = {spec.name for spec in skills_pkg.REGISTRY.values()}
    names.add(skills_pkg.DEFAULT_SKILL.name)
    assert {"table", "exhaustive", "comparison", "verification", "steps", "qa"} <= names
    assert skills_pkg.DEFAULT_SKILL.name == "qa"
    # Cada skill cargada tiene directivas no vacías (el cuerpo del .md).
    for spec in skills_pkg.REGISTRY.values():
        assert spec.directives.strip()


def test_frontmatter_parser():
    from retie_agent.agent.skills import _parse_frontmatter

    meta, body = _parse_frontmatter("---\nname: x\nmax_tokens: 42\n---\n\ncuerpo de la skill")
    assert meta == {"name": "x", "max_tokens": "42"}
    assert body == "cuerpo de la skill"
    # Sin frontmatter → todo es cuerpo.
    meta2, body2 = _parse_frontmatter("solo directivas")
    assert meta2 == {} and body2 == "solo directivas"


# ── Tools: acumulan evidencia y degradan sin lanzar ──────────────────────────

def test_search_chroma_acumula_en_log(monkeypatch):
    import retie_agent.retriever.retrieve as retrieve_mod

    monkeypatch.setattr(
        retrieve_mod, "search",
        lambda q, top_k=6, collection_name=None: _hits(),
    )
    log = EvidenceLog()
    tools = _build_tools(log, None)
    sc = next(t for t in tools if t.name == "search_chroma")

    out = sc.invoke({"query": "tabla 220.55"})
    assert "contenido del chunk" in out
    assert len(log.chroma_hits) == 1
    assert log.tool_calls == [
        {"tool": "search_chroma", "query": "tabla 220.55", "n_results": 1}
    ]


def test_search_chroma_error_degrada_sin_lanzar(monkeypatch):
    import retie_agent.retriever.retrieve as retrieve_mod

    def _boom(*a, **k):
        raise RuntimeError("chroma caído")

    monkeypatch.setattr(retrieve_mod, "search", _boom)
    log = EvidenceLog()
    sc = next(t for t in _build_tools(log, None) if t.name == "search_chroma")

    out = sc.invoke({"query": "x"})
    assert "falló" in out
    assert log.tool_calls[0]["error"].startswith("chroma caído")


def test_ask_gemini_solo_si_disponible(monkeypatch):
    monkeypatch.setattr(da, "_get_gemini", lambda: _FakeGemini(available=True))
    names = {t.name for t in _build_tools(EvidenceLog(), None)}
    assert "ask_gemini" in names

    monkeypatch.setattr(da, "_get_gemini", lambda: _FakeGemini(available=False))
    names = {t.name for t in _build_tools(EvidenceLog(), None)}
    assert "ask_gemini" not in names


def test_ask_gemini_acumula_fuentes(monkeypatch):
    monkeypatch.setattr(da, "_get_gemini", lambda: _FakeGemini(available=True))
    log = EvidenceLog()
    ag = next(t for t in _build_tools(log, None) if t.name == "ask_gemini")

    out = ag.invoke({"query": "puesta a tierra"})
    assert out == "respuesta gemini"
    assert len(log.gemini_sources) == 1
    assert log.tool_calls[0]["tool"] == "ask_gemini"


# ── Mensaje inicial: directivas del registry + parámetros del classifier ─────

def test_initial_message_skill_tabla_y_entidades():
    spec = resolve_skill("tabla", "markdown_table")
    msg = _build_initial_message(
        "dame la tabla 220.55",
        _hits(),
        "resumen de la fuente secundaria",
        spec,
        {
            "entities": [{"type": "tabla", "value": "220.55", "raw": "tabla 220.55"}],
            "complexity": "low",
            "needs_calculation": False,
            "confidence": 0.97,
        },
        wants_full=False,
    )
    assert "TODAS las filas" in msg                      # directiva de la skill
    assert "tabla 220.55" in msg                          # entidad como query candidata
    assert "ANÁLISIS COMPLEMENTARIO" in msg
    assert "PREGUNTA DEL USUARIO" in msg
    assert msg.index("FRAGMENTOS") < msg.index("DIRECTIVAS") < msg.index("PREGUNTA")


def test_initial_message_complejidad_y_calculo():
    msg = _build_initial_message(
        "pregunta compuesta", [], "", DEFAULT_SKILL,
        {"complexity": "high", "needs_calculation": True, "confidence": 0.3},
        wants_full=False,
    )
    assert "sub-preguntas" in msg
    assert "cálculo paso a paso" in msg
    assert "dudosa" in msg                                # confidence < 0.5
    assert "ninguno recuperado" in msg


def test_initial_message_sin_meta_solo_directiva_base():
    msg = _build_initial_message("q", [], "", DEFAULT_SKILL, None, wants_full=False)
    assert DEFAULT_SKILL.directives in msg
    assert "sub-preguntas" not in msg


# ── _chat_model_for: kwargs seguros por familia ───────────────────────────────

def test_chat_model_familia_razonadora_sin_temperature():
    m = _chat_model_for("gpt-5.4-nano-2026-03-17", 500)
    payload = m._get_request_payload([("user", "hola")])
    assert payload.get("max_completion_tokens") == 500
    assert "temperature" not in payload or payload["temperature"] is None


def test_chat_model_familia_clasica_con_temperature():
    m = _chat_model_for("gpt-4o-mini", 500)
    assert m.temperature == 0.0
    assert m.max_tokens == 500


# ── run_deep_answer: caminos ok / vacío / error / timeout ────────────────────

def test_run_deep_answer_ok(monkeypatch):
    import deepagents

    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    agent = _FakeAgent({"messages": [_ai("respuesta final", usage)]})
    monkeypatch.setattr(deepagents, "create_deep_agent", lambda **k: agent)

    r = run_deep_answer(
        "q", initial_hits=_hits(), secondary_answer="", history=[], agent_key=None
    )
    assert r.status == "ok"
    assert r.answer == "respuesta final"
    assert r.usage == {"input": 10, "output": 5, "total": 15}


def test_run_deep_answer_propaga_config_y_limites(monkeypatch):
    import deepagents

    agent = _FakeAgent({"messages": [_ai("ok")]})
    monkeypatch.setattr(deepagents, "create_deep_agent", lambda **k: agent)
    monkeypatch.setattr(da.settings, "DEEP_AGENT_RECURSION_LIMIT", 7, raising=False)

    run_deep_answer(
        "q", initial_hits=[], secondary_answer="", history=[],
        config={"callbacks": ["cb-langfuse"]},
    )
    assert agent.last_config["callbacks"] == ["cb-langfuse"]
    assert agent.last_config["recursion_limit"] == 7
    assert agent.last_config["run_name"] == "deep_answer_agent"


def test_run_deep_answer_respuesta_vacia_es_fallback(monkeypatch):
    import deepagents

    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **k: _FakeAgent({"messages": []})
    )
    r = run_deep_answer("q", initial_hits=[], secondary_answer="", history=[])
    assert r.status == "fallback_error"
    assert r.answer == ""


def test_run_deep_answer_excepcion_es_fallback(monkeypatch):
    import deepagents

    monkeypatch.setattr(
        deepagents, "create_deep_agent",
        lambda **k: _FakeAgent(exc=RuntimeError("boom")),
    )
    r = run_deep_answer("q", initial_hits=[], secondary_answer="", history=[])
    assert r.status == "fallback_error"


def test_run_deep_answer_entorno_roto_degrada_sin_lanzar(monkeypatch):
    """Regresión: un ImportError de deepagents (cluster de deps desalineado,
    p. ej. venv sin actualizar) debe degradar a fallback_error — jamás romper
    la respuesta al usuario (contrato 'nunca lanza')."""

    def _import_roto():
        raise ImportError("cannot import name 'ToolCallTransformer' from 'langgraph.prebuilt'")

    monkeypatch.setattr(da, "_ensure_harness_profile", _import_roto)
    r = run_deep_answer("q", initial_hits=_hits(), secondary_answer="", history=[])
    assert r.status == "fallback_error"
    assert r.answer == ""


def test_run_deep_answer_timeout(monkeypatch):
    import deepagents

    monkeypatch.setattr(
        deepagents, "create_deep_agent",
        lambda **k: _FakeAgent({"messages": [_ai("tarde")]}, delay=0.5),
    )
    monkeypatch.setattr(da.settings, "DEEP_AGENT_TIMEOUT", 0.05, raising=False)
    r = run_deep_answer("q", initial_hits=[], secondary_answer="", history=[])
    assert r.status == "fallback_timeout"
    assert r.answer == ""


# ── _dedupe_hits: inmune a scores None (hits de expansión de página) ─────────

def test_dedupe_hits_tolera_scores_none():
    """Regresión: expand_hits_with_page_context produce hits con score=None
    (page_expand). Dos None en la misma (source, page) daban
    "'<' not supported between instances of 'NoneType' and 'NoneType'"."""
    from retie_agent.agent.retie_agent import _dedupe_hits

    h_none_1 = {"text": "a", "score": None, "meta": {"source": "RETIE.pdf", "page": 5}}
    h_none_2 = {"text": "b", "score": None, "meta": {"source": "RETIE.pdf", "page": 5}}
    h_real = {"text": "c", "score": 0.2, "meta": {"source": "RETIE.pdf", "page": 5}}

    out = _dedupe_hits([h_none_1, h_none_2])  # no lanza
    assert len(out) == 1

    # Un score real siempre gana sobre None en la misma (source, page).
    out2 = _dedupe_hits([h_none_1, h_real, h_none_2])
    assert len(out2) == 1
    assert out2[0]["score"] == 0.2


def test_node_answer_une_hits_de_expansion_sin_lanzar(monkeypatch):
    """El caso real del bot: hits iniciales de página expandida (score=None) +
    hits acumulados por las tools del agente → el fan-out del nodo no revienta."""
    from retie_agent.agent import graph as g

    page_hits = [
        {"text": "fila 1", "score": None, "meta": {"source": "NTC.pdf", "page": 9}},
        {"text": "fila 2", "score": None, "meta": {"source": "NTC.pdf", "page": 9}},
    ]
    tool_hits = [
        {"text": "fila 3", "score": None, "meta": {"source": "NTC.pdf", "page": 9}},
        {"text": "otro", "score": 0.3, "meta": {"source": "NTC.pdf", "page": 10}},
    ]
    monkeypatch.setattr(
        g, "run_deep_answer", lambda *a, **k: _result("respuesta", tool_hits)
    )
    out = g._node_answer({
        "question": "dame la tabla 220.55",
        "chromadb_docs": page_hits,
        "wants_table": True,
    })
    assert out["route"] == "answer"
    assert len(out["hits"]) == 2  # (NTC.pdf, 9) deduplicada + (NTC.pdf, 10)


# ── _node_answer: contrato intocable y rutas conservadas ─────────────────────

def _result(answer="ok", hits=None, status="ok", skill="qa"):
    return DeepAnswerResult(
        answer=answer,
        hits=hits or [],
        status=status,
        tool_calls=[],
        usage={"input": None, "output": None, "total": None},
        skill=skill,
    )


def test_node_answer_contrato_de_claves(monkeypatch):
    from retie_agent.agent import graph as g

    tool_hits = _hits(source="NTC.pdf", page=3)
    monkeypatch.setattr(g, "run_deep_answer", lambda *a, **k: _result("respuesta", tool_hits))
    out = g._node_answer({
        "question": "q",
        "chromadb_docs": _hits(),
        "notebooklm_docs": "aporte nlm",
    })
    assert set(out) == {"answer", "hits", "nlm_answer", "route"}
    assert out["route"] == "answer"
    assert out["nlm_answer"] == "aporte nlm"
    # hits = iniciales + acumulados por tools, deduplicados por (source, page)
    assert len(out["hits"]) == 2


def test_node_answer_respeta_secondary_primary_gemini(monkeypatch):
    from retie_agent.agent import graph as g

    captured = {}

    def _spy(q, **kwargs):
        captured.update(kwargs)
        return _result("r")

    monkeypatch.setattr(g, "run_deep_answer", _spy)
    g._node_answer({
        "question": "q",
        "chromadb_docs": _hits(),
        "notebooklm_docs": "nlm",
        "gemini_docs": "gemini",
        "secondary_primary": "gemini",
    })
    assert captured["secondary_answer"] == "gemini"


def test_node_answer_fallback_degrada_a_sintesis_simple(monkeypatch):
    from retie_agent.agent import graph as g

    monkeypatch.setattr(
        g, "run_deep_answer", lambda *a, **k: _result("", status="fallback_timeout")
    )
    monkeypatch.setattr(
        g, "_synthesize_simple", lambda *a, **k: "respuesta del camino clásico"
    )
    out = g._node_answer({"question": "q", "chromadb_docs": _hits()})
    assert out["answer"] == "respuesta del camino clásico"
    assert out["route"] == "answer"


def test_node_answer_media_sin_agente(monkeypatch):
    from retie_agent.agent import graph as g

    def _no_agent(*a, **k):
        raise AssertionError("media_only no debe invocar al deep agent")

    monkeypatch.setattr(g, "run_deep_answer", _no_agent)
    seen = {}

    def _fake_simple(q, hits, nlm, history, model, wt, wf, *, media_only=False):
        seen["media_only"] = media_only
        return "interpretación de la imagen"

    monkeypatch.setattr(g, "_synthesize_simple", _fake_simple)
    out = g._node_answer({"question": "texto OCR de la foto", "source": "image"})
    assert seen["media_only"] is True
    assert out["route"] == "answer"


def test_node_answer_sin_evidencia_sin_tools_corta(monkeypatch):
    from retie_agent.agent import graph as g

    monkeypatch.setattr(g, "deep_tools_available", lambda: False)
    monkeypatch.setattr(
        g, "run_deep_answer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe correr")),
    )
    out = g._node_answer({"question": "q"})
    assert out["route"] == "no_context"
    assert out["answer"] == NO_EVIDENCE_PHRASE


def test_node_answer_frase_canonica_rutea_no_context(monkeypatch):
    from retie_agent.agent import graph as g

    monkeypatch.setattr(
        g, "run_deep_answer", lambda *a, **k: _result(NO_EVIDENCE_PHRASE)
    )
    out = g._node_answer({"question": "q", "chromadb_docs": _hits()})
    assert out["route"] == "no_context"


def test_node_answer_sin_evidencia_con_tools_corre_el_agente(monkeypatch):
    from retie_agent.agent import graph as g

    monkeypatch.setattr(g, "deep_tools_available", lambda: True)
    monkeypatch.setattr(
        g, "run_deep_answer",
        lambda *a, **k: _result("encontrado tras reformular", _hits()),
    )
    out = g._node_answer({"question": "q"})
    assert out["route"] == "answer"
    assert out["answer"] == "encontrado tras reformular"
    assert len(out["hits"]) == 1
