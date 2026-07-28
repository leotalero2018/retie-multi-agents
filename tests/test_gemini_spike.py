# tests/test_gemini_spike.py
"""Tests offline del spike Gemini File Search (sin red).

Cubren lo que falló en el piloto real:
- el ruteo del feature flag SECONDARY_RAG_SOURCE (quién corre y quién alimenta),
- la degradación del cliente cuando falta key/store,
- el reintento SOLO ante errores transitorios (503/429) — un 404 de modelo
  retirado NO debe reintentarse (ni colgar el nodo),
- la extracción defensiva de citas del grounding_metadata.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from retie_agent.agent.gemini_client import GeminiFileSearchClient, GeminiError
from retie_agent.config import settings
import retie_agent.agent.graph as graph


# ── Feature flag: _resolve_rag_sources ───────────────────────────────────────
# Devuelve (run_chroma, run_nlm, run_gemini, primary).

@pytest.fixture
def gemini_configured(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "GEMINI_FILE_SEARCH_STORE", "fileSearchStores/test")
    monkeypatch.setattr(settings, "NOTEBOOKLM_ENABLED", "true")


def test_flag_notebooklm_default(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "notebooklm")
    assert graph._resolve_rag_sources() == (True, True, False, "notebooklm")


def test_flag_gemini(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "gemini")
    assert graph._resolve_rag_sources() == (True, False, True, "gemini")


def test_flag_shadow_corre_los_tres_y_alimenta_nlm(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "shadow")
    assert graph._resolve_rag_sources() == (True, True, True, "notebooklm")


def test_flag_chroma_solo(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "chroma")
    assert graph._resolve_rag_sources() == (True, False, False, "notebooklm")


def test_flag_solo_gemini_sin_chroma(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "solo-gemini")
    assert graph._resolve_rag_sources() == (False, False, True, "gemini")


def test_flag_solo_notebooklm_sin_chroma(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "solo-notebooklm")
    assert graph._resolve_rag_sources() == (False, True, False, "notebooklm")


def test_flag_tolera_comillas_pegadas_de_railway(gemini_configured, monkeypatch):
    # Pegar `"shadow"` (con comillas) en el dashboard de Railway no debe caer al default.
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", '"shadow"')
    assert graph._resolve_rag_sources() == (True, True, True, "notebooklm")


def test_flag_gemini_sin_store_no_corre(monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "GEMINI_FILE_SEARCH_STORE", None)
    run_chroma, run_nlm, run_gemini, primary = graph._resolve_rag_sources()
    assert run_gemini is False
    assert primary == "gemini"  # answer_node leerá gemini_docs vacío y degrada a Chroma


def test_flag_desconocido_es_retrocompatible(gemini_configured, monkeypatch):
    monkeypatch.setattr(settings, "SECONDARY_RAG_SOURCE", "algo-raro")
    assert graph._resolve_rag_sources() == (True, True, False, "notebooklm")


# ── Cliente: disponibilidad y validación ─────────────────────────────────────

def test_is_available_requiere_key_y_store():
    assert not GeminiFileSearchClient(api_key=None, store=None).is_available()
    assert not GeminiFileSearchClient(api_key="k", store="").is_available()
    assert not GeminiFileSearchClient(api_key="", store="fileSearchStores/x").is_available()
    assert GeminiFileSearchClient(api_key="k", store="fileSearchStores/x").is_available()


def test_ask_question_sin_store_lanza_error():
    client = GeminiFileSearchClient(api_key="k", store=None)
    with pytest.raises(GeminiError, match="GEMINI_FILE_SEARCH_STORE"):
        client.ask_question("¿pregunta?")


# ── Reintentos: transitorios sí, permanentes no ──────────────────────────────

def _client_with_fake_sdk(side_effect):
    """Cliente con el SDK reemplazado: side_effect alimenta generate_content."""
    client = GeminiFileSearchClient(api_key="k", store="fileSearchStores/x")
    fake = MagicMock()
    fake.models.generate_content.side_effect = side_effect
    client._client = fake  # evita _ensure_client (y el import real del SDK)
    return client, fake


def _ok_response(text="respuesta"):
    return SimpleNamespace(text=text, candidates=[])


def test_404_modelo_retirado_no_se_reintenta(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client, fake = _client_with_fake_sdk(
        RuntimeError("404 NOT_FOUND. This model is no longer available to new users.")
    )
    with pytest.raises(GeminiError):
        client.ask_question("¿pregunta?")
    assert fake.models.generate_content.call_count == 1  # sin reintentos


def test_503_se_reintenta_y_recupera(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client, fake = _client_with_fake_sdk(
        [RuntimeError("503 UNAVAILABLE high demand"), _ok_response("ok tras retry")]
    )
    answer, sources = client.ask_question("¿pregunta?")
    assert answer == "ok tras retry"
    assert fake.models.generate_content.call_count == 2


def test_429_agota_reintentos_y_falla(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client, fake = _client_with_fake_sdk(RuntimeError("429 RESOURCE_EXHAUSTED quota"))
    with pytest.raises(GeminiError):
        client.ask_question("¿pregunta?")
    assert fake.models.generate_content.call_count == 4  # 1 + 3 reintentos


# ── Citas: extracción defensiva del grounding_metadata ───────────────────────

def test_extract_citations_shape_file_search():
    ctx = SimpleNamespace(
        title=None,
        document_name="4__Libro_3_-_Instalaciones",
        uri=None,
        page_number=40,
        text="Los valores máximos de resistencia de puesta a tierra...",
    )
    chunk = SimpleNamespace(retrieved_context=ctx, web=None)
    cand = SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=[chunk]))
    resp = SimpleNamespace(text="respuesta", candidates=[cand])

    sources = GeminiFileSearchClient._extract_citations(resp)
    assert sources == [
        {
            "source": "4__Libro_3_-_Instalaciones",
            "page": 40,
            "snippet": "Los valores máximos de resistencia de puesta a tierra...",
        }
    ]


def test_extract_citations_respuesta_sin_grounding_no_rompe():
    assert GeminiFileSearchClient._extract_citations(SimpleNamespace(text="x")) == []
    assert GeminiFileSearchClient._extract_citations(None) == []


def test_default_del_flag_es_gemini():
    """Cierre del piloto shadow (2026-07-27): Gemini pasa a alimentar la respuesta.

    Con "shadow" los 3 nodos corrían pero answer_node leía notebooklm_docs, así
    que la respuesta de Gemini nunca llegaba al agente (quedaba solo en su span).
    """
    from retie_agent.config import Settings

    assert Settings.model_fields["SECONDARY_RAG_SOURCE"].default == "gemini"
