# app/llm/chroma_stub.py

class FakeDoc:
    def __init__(self, page_content: str):
        self.page_content = page_content


class FakeVectorStore:
    def similarity_search(self, query: str, k: int = 3):
        # Devuelve siempre algunos resultados falsos
        return [
            FakeDoc(f"Documento 1 relevante para: {query}"),
            FakeDoc(f"Documento 2 relacionado con: {query}"),
            FakeDoc(f"Documento 3 que menciona: {query}"),
        ]


def get_vector_store():
    return FakeVectorStore()
