from app.retriever.retrieve import search

def test_query():
    results = search("¿Cuál es la normativa para instalaciones eléctricas en Colombia?", collection_name="minio_test")
    for r in results[:5]:
        print("➡", r["text"][:200], "...")
