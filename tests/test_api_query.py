import pytest
from unittest.mock import patch, MagicMock


MOCK_RESPONSE = {
    "formatted_response": "El RETIE es el Reglamento Técnico de Instalaciones Eléctricas.",
    "sources": [],
}


@pytest.fixture(scope="module")
def client():
    mongo_mock = MagicMock()
    mongo_mock.list_recent_files = MagicMock(return_value=[])
    mongo_mock.get_image_bytes = MagicMock(side_effect=FileNotFoundError)

    with patch("retie_agent.api.main.run_graph", return_value=MOCK_RESPONSE), \
         patch("retie_agent.api.main.list_recent_files", return_value=[]), \
         patch("retie_agent.api.main.get_image_bytes", side_effect=FileNotFoundError):
        from fastapi.testclient import TestClient
        from retie_agent.api.main import app
        yield TestClient(app)


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_query_returns_200(client):
    r = client.get("/query?q=¿Qué es el RETIE?")
    assert r.status_code == 200


def test_query_response_has_required_keys(client):
    r = client.get("/query?q=¿Qué es el RETIE?")
    data = r.json()
    assert "question" in data
    assert "answer" in data
    assert "session_id" in data


def test_query_answer_is_string(client):
    r = client.get("/query?q=instalaciones eléctricas")
    assert isinstance(r.json()["answer"], str)


def test_query_echoes_question(client):
    r = client.get("/query?q=tensión nominal")
    assert r.json()["question"] == "tensión nominal"


def test_query_missing_param_returns_422(client):
    r = client.get("/query")
    assert r.status_code == 422


def test_query_custom_session_id_is_returned(client):
    r = client.get("/query?q=test&session_id=sesion-abc")
    assert r.json()["session_id"] == "sesion-abc"


def test_query_default_session_id(client):
    r = client.get("/query?q=test")
    assert r.json()["session_id"] == "api_query"
