# app/bootstrap_preflight.py
import os, sys, chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def rag_preflight():
    if not os.getenv("SELFTEST_RAG_QUERY"):
        return  # desactivado
    persist = os.getenv("CHROMA_DIR", "./data/chroma_db")
    col_name = os.getenv("COLLECTION_NAME", "retie_docs")
    api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    thr = float(os.getenv("SELFTEST_THRESHOLD", "0.45"))
    need = int(os.getenv("SELFTEST_MIN_COUNT", "1"))

    c = chromadb.PersistentClient(path=persist)
    col = c.get_collection(col_name)
    ef = OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

    q = os.environ["SELFTEST_RAG_QUERY"]
    q_emb = ef([q])
    res = col.query(query_embeddings=q_emb, n_results=3, include=["documents","distances","metadatas"])
    dists = (res.get("distances") or [[]])[0]
    kept = [d for d in dists if d <= thr]
    print(f"[SELFTEST] COUNT={col.count()} dists={dists} kept<={thr}={len(kept)}")
    if len(kept) < need:
        print("[SELFTEST] FAILED: RAG no devuelve suficientes matches")
        # opcional: abortar
        # sys.exit(1)

def run():
    try:
        rag_preflight()
    except Exception as e:
        print("[SELFTEST] ERROR:", e)
        # sys.exit(1)

# Llama run() desde tu main antes de iniciar el bot
