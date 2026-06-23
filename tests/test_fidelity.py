"""Pruebas del verificador de fidelidad del enriquecimiento (TICKET-003 / H-903).

Garantía central: una respuesta con literales a)–f) y valores numéricos debe
sobrevivir intacta. Si el enriquecedor altera/omite/inventa cualquier número,
referencia o elemento de lista, la salida se descarta y se entrega el borrador.
"""
import pytest

from retie_agent.agent.fidelity import (
    fidelity_tokens,
    fidelity_diff,
    enrichment_preserves_fidelity,
)


# Borrador normativo realista: lista a)–f) + valores + referencias.
DRAFT = """El artículo 250.122 establece los siguientes requisitos:
a) conductor de cobre calibre 12 AWG
b) tensión nominal máxima de 120 V
c) corriente de 20 A por circuito
d) distancia mínima de 1,5 m a partes vivas
e) factor de demanda de 0.75
f) temperatura de operación de 75 °C
Ver además la Tabla 220.55 y el numeral 110.26(C)."""


def test_identical_text_preserves_fidelity():
    assert enrichment_preserves_fidelity(DRAFT, DRAFT) is True


def test_connective_rewrite_preserves_fidelity():
    """Reescritura solo de prosa conectiva: mismos números, refs y literales."""
    enriched = (
        "Conforme al artículo 250.122, se establecen estos requisitos. "
        "En primer lugar, a) conductor de cobre calibre 12 AWG; "
        "asimismo, b) tensión nominal máxima de 120 V; "
        "del mismo modo, c) corriente de 20 A por circuito; "
        "además, d) distancia mínima de 1,5 m a partes vivas; "
        "también, e) factor de demanda de 0.75; "
        "y finalmente, f) temperatura de operación de 75 °C. "
        "Conviene revisar la Tabla 220.55 y el numeral 110.26(C)."
    )
    assert enrichment_preserves_fidelity(DRAFT, enriched) is True


def test_dropped_list_item_is_rejected():
    """Se omite el literal d) → debe rechazarse."""
    enriched = DRAFT.replace("d) distancia mínima de 1,5 m a partes vivas\n", "")
    assert enrichment_preserves_fidelity(DRAFT, enriched) is False
    diff = fidelity_diff(DRAFT, enriched)
    assert diff["missing"]  # algo se perdió


def test_changed_numeric_value_is_rejected():
    """120 V → 110 V (alteración de valor) → debe rechazarse."""
    enriched = DRAFT.replace("120 V", "110 V")
    assert enrichment_preserves_fidelity(DRAFT, enriched) is False
    diff = fidelity_diff(DRAFT, enriched)
    assert "n:120" in diff["missing"]
    assert "n:110" in diff["added"]


def test_hallucinated_value_is_rejected():
    """Se inventa un valor no presente en el borrador → debe rechazarse."""
    enriched = DRAFT + "\nNota adicional: la resistencia máxima es de 25 ohm."
    assert enrichment_preserves_fidelity(DRAFT, enriched) is False
    assert "n:25" in fidelity_diff(DRAFT, enriched)["added"]


def test_changed_reference_is_rejected():
    """Cambia el número de tabla 220.55 → 220.56 → debe rechazarse."""
    enriched = DRAFT.replace("Tabla 220.55", "Tabla 220.56")
    assert enrichment_preserves_fidelity(DRAFT, enriched) is False


def test_full_list_letters_tracked():
    """Los 6 literales a–f quedan registrados como tokens de lista."""
    tokens = fidelity_tokens(DRAFT)
    for letter in "abcdef":
        assert tokens[f"l:{letter}"] >= 1
    # El literal inline (C) también se rastrea.
    assert tokens["l:c"] >= 1


def test_reordering_list_items_preserves_fidelity():
    """Reordenar literales mantiene el mismo multiconjunto → se acepta.

    (El prompt prohíbe reordenar, pero el verificador es por multiconjunto: si los
    tokens son los mismos, la fidelidad numérica/referencial está intacta.)"""
    enriched = """El artículo 250.122 establece los siguientes requisitos:
f) temperatura de operación de 75 °C
e) factor de demanda de 0.75
d) distancia mínima de 1,5 m a partes vivas
c) corriente de 20 A por circuito
b) tensión nominal máxima de 120 V
a) conductor de cobre calibre 12 AWG
Ver además la Tabla 220.55 y el numeral 110.26(C)."""
    assert enrichment_preserves_fidelity(DRAFT, enriched) is True


# ── Integración con el nodo del grafo ─────────────────────────────────────────

def test_enrich_node_rejects_unfaithful(monkeypatch):
    """El nodo descarta una salida que cambia un valor y omite un literal."""
    from retie_agent.agent import graph

    class FakeAgent:
        def enrich_response(self, user_message, draft_response):
            # Cambia 120→110 y elimina el literal d): debe ser rechazada.
            return draft_response.replace("120 V", "110 V").replace(
                "d) distancia mínima de 1,5 m a partes vivas\n", ""
            )

    monkeypatch.setattr(graph, "_get_enrichment_agent", lambda: FakeAgent())
    monkeypatch.setattr(graph.settings, "ENRICHMENT_ENABLED", "true")

    out = graph._node_enrich({"answer": DRAFT, "question": "requisitos del 250.122"})
    assert out["answer"] == DRAFT  # se entregó el borrador, no la versión alterada


def test_enrich_node_accepts_faithful(monkeypatch):
    """El nodo conserva un enriquecimiento que respeta todos los tokens críticos."""
    from retie_agent.agent import graph

    faithful = (
        "El artículo 250.122 establece los requisitos siguientes: "
        "a) conductor de cobre calibre 12 AWG, b) tensión nominal máxima de 120 V, "
        "c) corriente de 20 A por circuito, d) distancia mínima de 1,5 m a partes vivas, "
        "e) factor de demanda de 0.75, f) temperatura de operación de 75 °C. "
        "Ver además la Tabla 220.55 y el numeral 110.26(C)."
    )

    class FakeAgent:
        def enrich_response(self, user_message, draft_response):
            return faithful

    monkeypatch.setattr(graph, "_get_enrichment_agent", lambda: FakeAgent())
    monkeypatch.setattr(graph.settings, "ENRICHMENT_ENABLED", "true")

    out = graph._node_enrich({"answer": DRAFT, "question": "requisitos del 250.122"})
    assert out["answer"] == faithful
