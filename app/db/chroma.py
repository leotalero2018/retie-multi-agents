# app/db/chroma.py
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Usa las variables de entorno definidas en el Dockerfile o en Railway
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "retie_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_vector_store():
    """
    Inicializa y devuelve un vector store de Chroma.
    Si no hay documentos cargados aún, devolverá el store vacío.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    return vector_store
