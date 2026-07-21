"""Pruebas del léxico regex del clasificador (TICKET-001 / H-901).

El set parametrizado (≥30 formulaciones reales) corre SIN red: un fixture
autouse fuerza `_llm_classify_v3 -> None`, de modo que se ejerce el FALLBACK
regex ampliado del classifier v3 — que por sí solo debe clasificar
correctamente el léxico normativo del negocio (incluida la consulta canónica
"dame los requerimientos para X", el caso que H-901 fallaba).

El camino LLM, la cascada y los guardarraíles del v3 se prueban en
tests/test_intent_v3.py.
"""
import pytest

from retie_agent.agent import intent as intent_mod
from retie_agent.agent.intent import (
    classify_intent_v3,
    _wants_full,
    _wants_table,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Desactiva el clasificador LLM → todas las pruebas del set usan el regex."""
    monkeypatch.setattr(intent_mod, "_llm_classify_v3", lambda *a, **k: None)


# ── Set de formulaciones reales (≥30) ────────────────────────────────────────
# (consulta, intención esperada). Cubre las 4 categorías del léxico base.
CASES = [
    # — exhaustiva: léxico normativo del negocio (el caso que H-901 fallaba) —
    ("dame los requerimientos para puesta a tierra en zonas húmedas", "exhaustiva"),
    ("¿qué requisitos exige el RETIE para tableros eléctricos?", "exhaustiva"),
    ("lista completa de los numerales del artículo 250", "exhaustiva"),
    ("necesito todas las exigencias para instalaciones en piscinas", "exhaustiva"),
    ("¿qué condiciones establece la NTC 2050 para conductores en paralelo?", "exhaustiva"),
    ("enumera las obligaciones del instalador según el RETIE", "exhaustiva"),
    ("¿qué exige el artículo 110 sobre espacios de trabajo?", "exhaustiva"),
    ("quiero todos los literales del 250.122", "exhaustiva"),
    ("dame la respuesta completa sobre protección diferencial", "exhaustiva"),
    ("¿qué establece el RETIE para puestas a tierra de subestaciones?", "exhaustiva"),
    ("¿qué dice el artículo 225 sobre acometidas?", "exhaustiva"),
    ("requisitos de señalización para instalaciones eléctricas", "exhaustiva"),
    ("¿qué requiere la norma para canalizaciones subterráneas?", "exhaustiva"),
    ("dame la obligación de rotulado de tableros", "exhaustiva"),
    # — tabla —
    ("dame la tabla 220.55 de factores de demanda", "tabla"),
    ("muéstrame la tabla de calibres de conductores", "tabla"),
    ("¿hay una tabla con las distancias mínimas de seguridad?", "tabla"),
    ("tabla 310-15 de ampacidades", "tabla"),
    ("necesito la tabla de factores de corrección por temperatura", "tabla"),
    ("tablas de capacidad de corriente", "tabla"),
    ("la tabla completa 250.122 de conductores de puesta a tierra", "tabla"),
    # — comparativa (léxico nuevo del fallback v3) —
    ("diferencias entre RETIE y NTC 2050 en puesta a tierra", "comparativa"),
    ("cobre versus aluminio para acometidas", "comparativa"),
    # — puntual —
    ("¿cuál es la tensión nominal en Colombia?", "puntual"),
    ("¿qué significa GFCI?", "puntual"),
    ("¿a qué temperatura se prueba el aislamiento?", "puntual"),
    ("¿cuándo entró en vigencia el RETIE?", "puntual"),
    ("define acometida", "puntual"),
    ("¿qué calibre mínimo para un circuito de 20 amperios?", "puntual"),
    ("¿el neutro se considera conductor activo?", "puntual"),
    ("¿cuántos voltios se consideran peligrosos?", "puntual"),
    # — smalltalk —
    ("hola", "smalltalk"),
    ("muchas gracias", "smalltalk"),
    ("¿quién eres?", "smalltalk"),
    ("buenos días", "smalltalk"),
    ("perfecto", "smalltalk"),
]


def test_case_set_has_at_least_30_formulations():
    assert len(CASES) >= 30


@pytest.mark.parametrize("question,expected", CASES)
def test_regex_intent_classification(question, expected):
    result = classify_intent_v3(question)
    assert result.intent == expected, (
        f"{question!r} → {result.intent!r} (esperado {expected!r})"
    )


def test_h901_canonical_business_query_is_exhaustive():
    """La consulta canónica del negocio debe activar el camino completo."""
    result = classify_intent_v3("dame los requerimientos para X")
    assert result.intent == "exhaustiva"
    assert result.wants_full is True
    assert result.wants_table is False


# ── Mapeo de intención a flags / ruta ─────────────────────────────────────────

def test_flags_mapping():
    tabla = classify_intent_v3("dame la tabla 220.55")
    assert (tabla.wants_table, tabla.route) == (True, "retrieve")

    full = classify_intent_v3("dame todos los requisitos de puesta a tierra")
    assert (full.wants_table, full.wants_full, full.route) == (False, True, "retrieve")

    puntual = classify_intent_v3("¿cuál es la tensión nominal?")
    assert (puntual.wants_table, puntual.wants_full, puntual.route) == (False, False, "retrieve")

    chat = classify_intent_v3("hola")
    assert chat.route == "smalltalk"
    assert chat.requires_rag is False


# ── Compatibilidad de helpers ─────────────────────────────────────────────────

def test_backward_compat_helpers():
    assert _wants_full("dame los requisitos del artículo 250") is True
    assert _wants_full("¿cuál es la tensión nominal?") is False
    assert _wants_table("muéstrame la tabla 5") is True
    assert _wants_table("dame el requisito 250.122") is False
