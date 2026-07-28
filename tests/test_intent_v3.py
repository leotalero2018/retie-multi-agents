# tests/test_intent_v3.py
"""Tests del classifier v3 (classifier_node) — FINAL_IMPLEMENTATION_NODES §4.

v3 es el clasificador ÚNICO del grafo (sin flag). Sin red: el LLM se mockea vía
monkeypatch de intent_mod._llm_classify_v3. Cubre: extracción de entidades,
parser tolerante, guardarraíles, cascada de modelos, reglas duras de derivación
y el wiring nuevo del grafo. El léxico del fallback regex se prueba en
tests/test_intent.py.
"""
import pytest

from retie_agent.agent import intent as intent_mod
from retie_agent.agent.intent import (
    IntentResultV3,
    classify_intent_v3,
    extract_entities,
    _parse_intent_json,
)


def _llm_data(**overrides):
    """Respuesta base del classifier LLM v3 (dict ya parseado)."""
    data = {
        "intent": "puntual",
        "response_format": "markdown",
        "output_length": "medium",
        "complexity": "low",
        "needs_calculation": False,
        "confidence": 0.9,
    }
    data.update(overrides)
    return data


@pytest.fixture
def llm(monkeypatch):
    """Fija la salida del LLM v3 y cuenta llamadas por modelo."""
    calls = []

    def _set(data_or_fn):
        def _fake(question, model):
            calls.append(model)
            if callable(data_or_fn):
                return data_or_fn(question, model)
            return dict(data_or_fn) if data_or_fn is not None else None

        monkeypatch.setattr(intent_mod, "_llm_classify_v3", _fake)
        return calls

    return _set


# ── Extracción de entidades (determinística) ─────────────────────────────────

def test_entities_tabla_sin_ref_duplicada():
    ents = extract_entities("dame la tabla 220.55 de factores de demanda")
    assert {"type": "tabla", "value": "220.55", "raw": "tabla 220.55"} in ents
    assert not any(e["type"] == "ref" and e["value"] == "220.55" for e in ents)


def test_entities_articulo_y_norma():
    ents = extract_entities("¿qué dice el artículo 20.23 del RETIE?")
    types = {(e["type"], e["value"]) for e in ents}
    assert ("articulo", "20.23") in types
    assert ("norma", "RETIE") in types


def test_entities_ref_suelta():
    ents = extract_entities("explícame la 110-14 por favor")
    assert {"type": "ref", "value": "110-14", "raw": "110-14"} in ents


def test_entities_dedup_y_ntc():
    ents = extract_entities("tabla 220.55 y otra vez la tabla 220.55 de la NTC 2050")
    tablas = [e for e in ents if e["type"] == "tabla"]
    assert len(tablas) == 1
    assert any(e["type"] == "norma" and e["value"] == "NTC 2050" for e in ents)


def test_entities_texto_sin_referencias():
    assert extract_entities("requisitos de puesta a tierra") == []


def test_entities_tabla_con_sufijo_retie():
    """Los IDs RETIE llevan sufijo .a/.b: debe capturarse completo (el lookup
    canónico depende de él)."""
    ents = extract_entities("dame la tabla 2.3.26.2.2.1.a de galvanizado")
    assert any(e["type"] == "tabla" and e["value"] == "2.3.26.2.2.1.a" for e in ents)


def test_entities_tabla_con_sufijo_ntc():
    ents = extract_entities("muéstrame la tabla 392.10(A) completa")
    assert any(e["type"] == "tabla" and e["value"] == "392.10(A)" for e in ents)
    # Con espacio antes del paréntesis, normaliza al mismo id.
    ents2 = extract_entities("la tabla 392.10 (A) por favor")
    assert any(e["type"] == "tabla" and e["value"] == "392.10(A)" for e in ents2)


def test_entities_tabla_sin_sufijo_no_absorbe_titulo():
    ents = extract_entities("dame la tabla 220.55. Factores de demanda")
    vals = [e["value"] for e in ents if e["type"] == "tabla"]
    assert vals == ["220.55"]


# ── Parser tolerante ──────────────────────────────────────────────────────────

def test_parse_json_valido():
    raw = (
        '{"intent": "tabla", "response_format": "markdown_table", "output_length":'
        ' "detailed", "complexity": "low", "needs_calculation": false, "confidence": 0.97}'
    )
    data = _parse_intent_json(raw)
    assert data["intent"] == "tabla"
    assert data["response_format"] == "markdown_table"
    assert data["confidence"] == 0.97


def test_parse_json_con_fences_y_texto():
    raw = 'Claro:\n```json\n{"intent": "puntual", "confidence": 0.8}\n```'
    data = _parse_intent_json(raw)
    assert data["intent"] == "puntual"
    # Campos ausentes degradan a defaults seguros.
    assert data["response_format"] == "markdown"
    assert data["output_length"] == "medium"


def test_parse_intent_invalido_devuelve_none():
    assert _parse_intent_json('{"intent": "poema", "confidence": 0.9}') is None
    assert _parse_intent_json("no soy json") is None
    assert _parse_intent_json("") is None


def test_parse_confidence_se_normaliza():
    assert _parse_intent_json('{"intent": "puntual", "confidence": 3.7}')["confidence"] == 1.0
    assert _parse_intent_json('{"intent": "puntual", "confidence": -1}')["confidence"] == 0.0
    # confidence no numérica → 0.5 (neutral)
    assert _parse_intent_json('{"intent": "puntual", "confidence": "alta"}')["confidence"] == 0.5


# ── Fast-paths regex (nunca pagan LLM) ────────────────────────────────────────

def test_saludo_regex_no_llama_llm(llm):
    calls = llm(_llm_data())
    res = classify_intent_v3("hola, buenas tardes")
    assert (res.intent, res.route, res.requires_rag) == ("smalltalk", "smalltalk", False)
    assert res.source == "regex" and res.confidence == 1.0
    assert calls == []


@pytest.mark.parametrize("q", ["graciaaas!!", "holaaaa", "muchas graciasss", "heyyy, graciaaas"])
def test_saludo_alargado_es_smalltalk_sin_llm(llm, q):
    """Alargamientos expresivos: sin la normalización, "graciaaas" se salía del
    fast-path y el guardarraíl 5a lo degradaba a puntual → RAG completo."""
    calls = llm(_llm_data(intent="puntual"))
    res = classify_intent_v3(q)
    assert (res.intent, res.route, res.requires_rag) == ("smalltalk", "smalltalk", False)
    assert calls == []


def test_alargamiento_no_convierte_consulta_en_saludo(llm):
    # La normalización no debe volver smalltalk una consulta real que arranca
    # con saludo alargado + pregunta: manda la pregunta.
    llm(_llm_data(intent="puntual", confidence=0.9))
    res = classify_intent_v3("holaaaa, ¿qué es una acometida?")
    assert res.intent == "puntual"
    assert res.route == "retrieve"


def test_fragmento_vago_sin_historial_es_ambiguous(llm):
    calls = llm(_llm_data())
    res = classify_intent_v3("que es")
    assert (res.intent, res.route, res.requires_rag) == ("ambiguous", "ambiguous", False)
    assert calls == []


def test_media_nunca_es_smalltalk_ni_vago(llm):
    llm(_llm_data(intent="puntual"))
    res = classify_intent_v3("hola", is_media=True)
    assert res.route == "retrieve"
    assert res.intent == "puntual"


# ── Guardarraíles ─────────────────────────────────────────────────────────────

def test_llm_smalltalk_sin_confirmacion_regex_se_reclasifica(llm):
    llm(_llm_data(intent="smalltalk", confidence=0.9))
    res = classify_intent_v3("explícame el RETIE completo por favor")
    assert res.intent != "smalltalk"
    assert res.route == "retrieve"
    assert res.source.endswith("+guard")


def test_fuera_de_dominio_alta_confianza_corta(llm):
    llm(_llm_data(intent="fuera_de_dominio", confidence=0.97))
    res = classify_intent_v3("recomiéndame una película de acción")
    assert res.route == "out_of_domain"
    assert res.requires_rag is False


def test_fuera_de_dominio_baja_confianza_degrada_a_retrieve(llm):
    llm(_llm_data(intent="fuera_de_dominio", confidence=0.5))
    res = classify_intent_v3("¿el aluminio sirve para eso?")
    assert res.route == "retrieve"
    assert res.intent == "puntual"
    assert res.source.endswith("+guard")
    assert res.requires_rag is True


def test_fuera_de_dominio_nunca_por_regex(llm):
    # LLM caído → el fallback regex jamás emite fuera_de_dominio.
    llm(None)
    res = classify_intent_v3("recomiéndame una película de acción")
    assert res.route == "retrieve"
    assert res.source == "regex"


def test_fuera_de_dominio_desde_imagen_va_a_image_direct(llm):
    """Guardarraíl 5c: una consulta que viene de imagen nunca recibe el rechazo
    fijo — el contenido leído (OCR/Vision) viaja en la pregunta y suele contener
    la respuesta (p. ej. la clase de una etiqueta de eficiencia)."""
    llm(_llm_data(intent="fuera_de_dominio", confidence=0.97))
    res = classify_intent_v3(
        "Usuario dijo sobre la imagen: ¿cuál es la etiqueta de eficiencia de esta lavadora?\n\n"
        "Contenido interpretado de la imagen:\nEtiqueta de consumo de energía, clase A…",
        media_source="image",
    )
    assert res.route == "image_direct"
    assert res.requires_rag is False
    assert res.source.endswith("+guard")


def test_fuera_de_dominio_desde_voz_conserva_el_corte(llm):
    # La ruta image_direct es SOLO para imágenes: una nota de voz fuera de
    # dominio mantiene el mensaje fijo (evita usar voz para saltarse el dominio).
    llm(_llm_data(intent="fuera_de_dominio", confidence=0.97))
    res = classify_intent_v3("recomiéndame una serie", media_source="voice")
    assert res.route == "out_of_domain"


def test_imagen_en_dominio_sigue_a_retrieve(llm):
    # Una imagen con pregunta eléctrica clasifica normal → RAG completo.
    llm(_llm_data(intent="verificacion", confidence=0.9))
    res = classify_intent_v3(
        "Usuario dijo sobre la imagen: ¿este tablero cumple el RETIE?",
        media_source="image",
    )
    assert res.route == "retrieve"
    assert res.requires_rag is True


# ── Fallback regex v3 ─────────────────────────────────────────────────────────

def test_regex_fallback_tabla(llm):
    llm(None)
    res = classify_intent_v3("dame la tabla 220.55")
    assert (res.intent, res.source) == ("tabla", "regex")
    assert res.wants_table is True
    assert res.response_format == "markdown_table"
    assert res.confidence == 0.6


def test_regex_fallback_comparativa(llm):
    llm(None)
    res = classify_intent_v3("diferencias entre RETIE y NTC 2050 en puesta a tierra")
    assert res.intent == "comparativa"
    assert res.wants_full is True
    assert res.response_format == "comparison"


def test_regex_fallback_exhaustiva(llm):
    llm(None)
    res = classify_intent_v3("dame los requisitos de puesta a tierra")
    assert res.intent == "exhaustiva"
    assert res.wants_full is True


# ── Reglas duras de derivación ────────────────────────────────────────────────

def test_wants_table_por_response_format(llm):
    llm(_llm_data(intent="puntual", response_format="markdown_table"))
    res = classify_intent_v3("¿cuántos kW usa una estufa según la 220.55?")
    assert res.wants_table is True


def test_exhaustiva_fuerza_output_detailed(llm):
    llm(_llm_data(intent="exhaustiva", output_length="short"))
    res = classify_intent_v3("requisitos del artículo 250")
    assert res.wants_full is True
    assert res.output_length == "detailed"


def test_requires_rag_derivado_de_route(llm):
    llm(_llm_data(intent="verificacion", confidence=0.9))
    res = classify_intent_v3("¿puedo usar calibre 14 en tomas de 20 A?")
    assert (res.route, res.requires_rag) == ("retrieve", True)


# ── Cascada de modelos ────────────────────────────────────────────────────────

def test_cascada_escala_con_baja_confianza(llm, monkeypatch):
    monkeypatch.setattr(intent_mod.settings, "INTENT_V3_ESCALATION_MODEL", "modelo-potente", raising=False)

    def _por_modelo(question, model):
        if model == "modelo-potente":
            return _llm_data(intent="comparativa", confidence=0.95)
        return _llm_data(intent="puntual", confidence=0.4)

    calls = llm(_por_modelo)
    res = classify_intent_v3("cobre o aluminio para acometidas?")
    assert res.source == "llm+escalated"
    assert res.intent == "comparativa"
    assert len(calls) == 2 and calls[1] == "modelo-potente"


def test_cascada_intent_sensible_escala_bajo_085(llm, monkeypatch):
    monkeypatch.setattr(intent_mod.settings, "INTENT_V3_ESCALATION_MODEL", "modelo-potente", raising=False)
    calls = llm(lambda q, m: _llm_data(intent="fuera_de_dominio", confidence=0.8))
    res = classify_intent_v3("háblame de paneles solares en Perú")
    assert len(calls) == 2  # 0.8 < INTENT_V3_SENSITIVE_CONF (0.85) → escala


def test_cascada_no_escala_con_confianza_alta(llm, monkeypatch):
    monkeypatch.setattr(intent_mod.settings, "INTENT_V3_ESCALATION_MODEL", "modelo-potente", raising=False)
    calls = llm(_llm_data(intent="puntual", confidence=0.95))
    classify_intent_v3("¿qué es una acometida?")
    assert len(calls) == 1


def test_sin_modelo_de_escalado_no_hay_cascada(llm, monkeypatch):
    monkeypatch.setattr(intent_mod.settings, "INTENT_V3_ESCALATION_MODEL", None, raising=False)
    calls = llm(_llm_data(intent="puntual", confidence=0.1))
    classify_intent_v3("¿qué es una acometida?")
    assert len(calls) == 1


# ── Serialización al estado del grafo ────────────────────────────────────────

def test_intent_llm_disabled_usa_regex(monkeypatch):
    # Sin flag de grafo: v3 es el único clasificador; INTENT_LLM_ENABLED=false
    # debe dejar todo el flujo funcionando solo con el regex (sin red).
    monkeypatch.setattr(intent_mod.settings, "INTENT_LLM_ENABLED", "false", raising=False)
    res = classify_intent_v3("dame la tabla 220.55")
    assert (res.intent, res.source, res.wants_table) == ("tabla", "regex", True)


def test_to_state_meta_es_json_safe(llm):
    import json as _json

    llm(None)  # regex fallback, sin red
    res = classify_intent_v3("dame la tabla 220.55")
    meta = res.to_state_meta()
    _json.dumps(meta)  # no lanza
    assert set(meta) == {
        "requires_rag", "response_format", "output_length", "complexity",
        "needs_calculation", "confidence", "entities",
    }


# ── Wiring del grafo ──────────────────────────────────────────────────────────

def test_node_classifier_publica_estado(monkeypatch):
    from retie_agent.agent import graph as graph_mod

    fake = IntentResultV3(
        intent="tabla", source="llm", route="retrieve",
        wants_table=True, wants_full=False, requires_rag=True,
        response_format="markdown_table", output_length="detailed",
        complexity="low", needs_calculation=False, confidence=0.97,
        entities=[{"type": "tabla", "value": "220.55", "raw": "tabla 220.55"}],
    )
    monkeypatch.setattr(graph_mod, "classify_intent_v3", lambda *a, **k: fake)
    out = graph_mod._node_classifier({"question": "dame la tabla 220.55"})
    assert out["route"] == "retrieve"
    assert out["intent"] == "tabla"
    assert out["wants_table"] is True
    assert out["intent_meta"]["response_format"] == "markdown_table"
    assert out["intent_meta"]["entities"][0]["value"] == "220.55"


def test_node_route_entry_ya_no_clasifica():
    from retie_agent.agent import graph as graph_mod

    out = graph_mod._node_route_entry({"question": "dame la tabla 220.55"})
    assert "intent" not in out and "route" not in out
    assert {"run_chromadb", "run_notebooklm", "run_gemini", "secondary_primary"} <= set(out)


def test_node_out_of_domain_mensaje_fijo():
    from retie_agent.agent import graph as graph_mod

    out = graph_mod._node_out_of_domain({"question": "recomiéndame una serie"})
    assert "RETIE" in out["answer"]


def test_node_smalltalk_gracias_alargado_agradece():
    # Un "gracias" alargado debe recibir el mensaje de agradecimiento,
    # no el saludo de bienvenida.
    from retie_agent.agent import graph as graph_mod

    out = graph_mod._node_smalltalk({"question": "graciaaas!!"})
    assert out["answer"] == graph_mod._SMALLTALK_THANKS


def test_node_image_answer_responde_desde_contenido(monkeypatch):
    from retie_agent.agent import graph as graph_mod

    seen = {}

    def _fake_synth(q, hits, nlm, history, model, wt, wf, *, media_only=False):
        seen.update({"media_only": media_only, "hits": hits})
        return "La etiqueta indica clase A. Ese etiquetado lo regula el RETIQ."

    monkeypatch.setattr(graph_mod, "_synthesize_simple", _fake_synth)
    out = graph_mod._node_image_answer(
        {"question": "Usuario dijo sobre la imagen: ¿cuál es la etiqueta de eficiencia?"}
    )
    assert "clase A" in out["answer"]
    # Sin RAG y con el prompt de media (el contenido ya viaja en la pregunta).
    assert seen == {"media_only": True, "hits": []}


def test_node_image_answer_fallback_si_llm_falla(monkeypatch):
    from retie_agent.agent import graph as graph_mod

    def _boom(*a, **k):
        raise RuntimeError("LLM caído")

    monkeypatch.setattr(graph_mod, "_synthesize_simple", _boom)
    out = graph_mod._node_image_answer({"question": "¿qué dice la etiqueta?"})
    # Red de seguridad: el mensaje fijo de fuera de dominio (comportamiento previo).
    assert out["answer"] == graph_mod._OUT_OF_DOMAIN_MSG


def test_node_classifier_propaga_media_source(monkeypatch):
    from retie_agent.agent import graph as graph_mod

    captured = {}

    def _fake_classify(q, **kwargs):
        captured.update(kwargs)
        return IntentResultV3(
            intent="fuera_de_dominio", source="llm+guard", route="image_direct",
            wants_table=False, wants_full=False, requires_rag=False,
            response_format="markdown", output_length="short",
            complexity="low", needs_calculation=False, confidence=0.97,
        )

    monkeypatch.setattr(graph_mod, "classify_intent_v3", _fake_classify)
    out = graph_mod._node_classifier({"question": "…", "source": "image"})
    assert captured["media_source"] == "image"
    assert captured["is_media"] is True
    assert out["route"] == "image_direct"


def test_grafo_compila_con_classifier_como_entry():
    from retie_agent.agent import graph as graph_mod

    app = graph_mod.build_graph()
    drawable = app.get_graph()
    nodes = set(drawable.nodes)
    assert {"classifier_node", "route_entry", "out_of_domain_node",
            "image_answer_node", "query_enrichment_node"} <= nodes
    assert "condense_node" not in nodes
    # image_answer_node desemboca en el stylist como los demás short-circuits.
    assert any(
        e.source == "image_answer_node" and e.target == "stylist_node"
        for e in drawable.edges
    )
    assert any(
        e.source == "__start__" and e.target == "classifier_node"
        for e in drawable.edges
    )


# ── Guardarraíl con evidencia sobre fuera_de_dominio ─────────────────────────
# El clasificador solo conoce la DESCRIPCIÓN del dominio; el índice conoce su
# CONTENIDO. Antes de cortar se verifica contra Chroma: preguntas como
# "¿en qué consiste la fibrilación ventricular?" viven en el Anexo General y
# sonaban a otra disciplina, así que el LLM las cortaba con confianza alta.

def _ood_result():
    return IntentResultV3(
        intent="fuera_de_dominio", source="llm", route="out_of_domain",
        wants_table=False, wants_full=False, requires_rag=False,
        response_format="markdown", output_length="short",
        complexity="low", needs_calculation=False, confidence=0.95,
    )


@pytest.fixture
def ood_graph(monkeypatch):
    """classifier_node con el LLM fijado en fuera_de_dominio y Chroma mockeable."""
    from retie_agent.agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "classify_intent_v3", lambda *a, **k: _ood_result())
    monkeypatch.setattr(graph_mod, "_resolve_collection", lambda *a, **k: "normativas")
    monkeypatch.setattr(graph_mod.settings, "OOD_EVIDENCE_GUARD", "true", raising=False)
    return graph_mod


def _dense_hit():
    return [{"text": "La fibrilación ventricular es…", "score": 0.21,
             "score_type": "cosine_distance", "meta": {"source": "RETIE.pdf", "page": 12}}]


def test_ood_guard_degrada_a_retrieve_cuando_el_indice_tiene_evidencia(ood_graph, monkeypatch):
    monkeypatch.setattr(ood_graph, "search", lambda *a, **k: _dense_hit())
    out = ood_graph._node_classifier({"question": "¿en qué consiste la fibrilación ventricular?"})
    assert out["route"] == "retrieve"
    assert out["intent"] == "puntual"
    assert out["intent_source"].endswith("+evidence_guard")
    assert out["intent_meta"]["requires_rag"] is True


def test_ood_guard_conserva_el_corte_sin_evidencia(ood_graph, monkeypatch):
    monkeypatch.setattr(ood_graph, "search", lambda *a, **k: [])
    out = ood_graph._node_classifier({"question": "recomiéndame una serie"})
    assert out["route"] == "out_of_domain"
    assert out["intent"] == "fuera_de_dominio"


def test_ood_guard_ignora_hits_solo_lexicos(ood_graph, monkeypatch):
    # BM25 matchea por palabras compartidas con cualquier pregunta: no es evidencia.
    bm25 = [{"text": "…", "score": 3.1, "score_type": "bm25", "meta": {}}]
    monkeypatch.setattr(ood_graph, "search", lambda *a, **k: bm25)
    out = ood_graph._node_classifier({"question": "cuál es la mejor receta de arroz"})
    assert out["route"] == "out_of_domain"


def test_ood_guard_desactivable_por_flag(ood_graph, monkeypatch):
    def _no_search(*a, **k):
        raise AssertionError("con el flag apagado no debe consultarse el índice")

    monkeypatch.setattr(ood_graph.settings, "OOD_EVIDENCE_GUARD", "false", raising=False)
    monkeypatch.setattr(ood_graph, "search", _no_search)
    out = ood_graph._node_classifier({"question": "recomiéndame una serie"})
    assert out["route"] == "out_of_domain"


def test_ood_guard_no_gasta_retrieval_si_la_consulta_es_del_dominio(monkeypatch):
    # Solo fuera_de_dominio paga la verificación; el resto no cambia de costo.
    from retie_agent.agent import graph as graph_mod

    fake = IntentResultV3(
        intent="puntual", source="llm", route="retrieve",
        wants_table=False, wants_full=False, requires_rag=True,
        response_format="markdown", output_length="short",
        complexity="low", needs_calculation=False, confidence=0.95,
    )
    monkeypatch.setattr(graph_mod, "classify_intent_v3", lambda *a, **k: fake)

    def _no_search(*a, **k):
        raise AssertionError("el guardarraíl no debe correr fuera del corte")

    monkeypatch.setattr(graph_mod, "search", _no_search)
    assert graph_mod._node_classifier({"question": "¿qué es el RETIE?"})["route"] == "retrieve"


def test_ood_guard_si_falla_el_indice_conserva_el_corte(ood_graph, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("Chroma caído")

    monkeypatch.setattr(ood_graph, "search", _boom)
    out = ood_graph._node_classifier({"question": "recomiéndame una serie"})
    assert out["route"] == "out_of_domain"


def test_mensaje_fuera_de_dominio_acota_a_los_libros(ood_graph):
    msg = ood_graph._OUT_OF_DOMAIN_MSG
    # No es el saludo de bienvenida: declara el alcance documental.
    assert "RETIE" in msg and "NTC 2050" in msg
    assert "Solo puedo responder" in msg
    assert "Soy un asistente especializado" not in msg
