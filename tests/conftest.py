import os
import pytest
import chromadb
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-000")
os.environ.setdefault("LANGFUSE_ENABLED", "false")
os.environ.setdefault("MINIO_PUBLIC_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ROOT_USER", "minioadmin")
os.environ.setdefault("MINIO_ROOT_PASSWORD", "minioadmin")
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017/")
os.environ.setdefault("ALLOW_CHROMA_CREATE", "true")
os.environ.setdefault("RAG_DISTANCE_THRESHOLD", "1.0")

FAKE_EMBEDDING = [0.1] * 1536

FAKE_DOCS = [
    "El RETIE regula las instalaciones eléctricas en Colombia",
    "Los circuitos deben tener protección contra sobrecorriente",
    "La tensión nominal en Colombia es 120V/60Hz",
]

FAKE_GRAPH_RESPONSE = {
    "formatted_response": "Respuesta de prueba sobre el RETIE.",
    "sources": [],
}


@pytest.fixture(scope="function")
def chroma_ephemeral():
    client = chromadb.EphemeralClient()
    col = client.create_collection("retie_docs", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=["d1", "d2", "d3"],
        documents=FAKE_DOCS,
        embeddings=[[0.10 + i * 0.05] * 1536 for i in range(3)],
        metadatas=[
            {"source": "retie.pdf", "page": 1},
            {"source": "retie.pdf", "page": 2},
            {"source": "retie.pdf", "page": 3},
        ],
    )
    return client, col


@pytest.fixture
def patch_embedder(monkeypatch):
    monkeypatch.setattr(
        "retie_agent.llm.embedder.embed_texts",
        lambda texts, **kw: [FAKE_EMBEDDING for _ in texts],
    )
    monkeypatch.setattr(
        "retie_agent.retriever.retrieve._embed_query_cached",
        lambda query: FAKE_EMBEDDING,
    )


@pytest.fixture
def patch_run_graph(monkeypatch):
    monkeypatch.setattr(
        "retie_agent.api.main.run_graph",
        lambda *a, **kw: FAKE_GRAPH_RESPONSE,
    )


@pytest.fixture
def mock_minio_client():
    mock = MagicMock()
    mock.bucket_exists.return_value = True
    mock.list_objects.return_value = []
    return mock
