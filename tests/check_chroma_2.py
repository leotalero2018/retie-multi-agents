import os, chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

persist = os.getenv("CHROMA_DIR", "./data/chroma_db")
col_name = os.getenv("COLLECTION_NAME", "retie_docs")
api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

c = chromadb.PersistentClient(path=persist)
col = c.get_collection(col_name)

ef = OpenAIEmbeddingFunction(api_key=api_key, model_name=model)
q = "dame el resumen de las Aplicaciones de redes neuronales en diagnóstico médico"
q_emb = ef([q])

res = col.query(query_embeddings=q_emb, n_results=3, include=["documents","distances","metadatas"])
print(res)
