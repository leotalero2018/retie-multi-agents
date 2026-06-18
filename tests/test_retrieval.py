import pytest
import chromadb
from unittest.mock import patch

FAKE_EMBEDDING = [0.1] * 1536


@pytest.fixture
def populated_collection():
    client = chromadb.EphemeralClient()
    col = client.create_collection("retie_docs", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=["d1", "d2", "d3"],
        documents=[
            "El RETIE regula instalaciones eléctricas en Colombia",
            "Los circuitos deben tener protección contra sobrecorriente",
            "Tensión nominal en Colombia es 120V/60Hz",
        ],
        embeddings=[
            [0.10] * 1536,
            [0.20] * 1536,
            [0.30] * 1536,
        ],
        metadatas=[
            {"source": "retie.pdf", "page": 1},
            {"source": "retie.pdf", "page": 2},
            {"source": "retie.pdf", "page": 3},
        ],
    )
    return client, col


def test_search_returns_list(populated_collection):
    _, col = populated_collection
    with patch("retie_agent.retriever.retrieve.get_collection", return_value=col), \
         patch("retie_agent.retriever.retrieve._embed_query_cached", return_value=FAKE_EMBEDDING):
        from retie_agent.retriever.retrieve import search
        results = search("instalaciones eléctricas", collection_name="retie_docs", top_k=3)
    assert isinstance(results, list)


def test_search_hits_have_text_key(populated_collection):
    _, col = populated_collection
    with patch("retie_agent.retriever.retrieve.get_collection", return_value=col), \
         patch("retie_agent.retriever.retrieve._embed_query_cached", return_value=FAKE_EMBEDDING):
        from retie_agent.retriever.retrieve import search
        results = search("circuitos protección", collection_name="retie_docs", top_k=3)
    assert len(results) > 0
    for hit in results:
        assert "text" in hit


def test_search_hits_have_score(populated_collection):
    _, col = populated_collection
    with patch("retie_agent.retriever.retrieve.get_collection", return_value=col), \
         patch("retie_agent.retriever.retrieve._embed_query_cached", return_value=FAKE_EMBEDDING):
        from retie_agent.retriever.retrieve import search
        results = search("RETIE Colombia", collection_name="retie_docs", top_k=3)
    assert len(results) > 0
    for hit in results:
        assert "score" in hit
        assert isinstance(hit["score"], float)


def test_search_respects_top_k(populated_collection):
    _, col = populated_collection
    with patch("retie_agent.retriever.retrieve.get_collection", return_value=col), \
         patch("retie_agent.retriever.retrieve._embed_query_cached", return_value=FAKE_EMBEDDING):
        from retie_agent.retriever.retrieve import search
        results = search("RETIE Colombia", collection_name="retie_docs", top_k=2)
    assert len(results) <= 2


def test_search_empty_collection_returns_list():
    client = chromadb.EphemeralClient()
    empty_col = client.create_collection("empty_col")
    with patch("retie_agent.retriever.retrieve.get_collection", return_value=empty_col), \
         patch("retie_agent.retriever.retrieve._embed_query_cached", return_value=FAKE_EMBEDDING):
        from retie_agent.retriever.retrieve import search
        results = search("algo que no existe", collection_name="empty_col", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 0


def test_search_missing_collection_returns_empty():
    with patch("retie_agent.retriever.retrieve.get_collection", side_effect=RuntimeError("collection not found")), \
         patch("retie_agent.retriever.retrieve._embed_query_cached", return_value=FAKE_EMBEDDING):
        from retie_agent.retriever.retrieve import search
        results = search("cualquier cosa", collection_name="inexistente", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 0
