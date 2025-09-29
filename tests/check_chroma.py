import os, chromadb
persist = os.getenv("CHROMA_DIR", "./data/chroma_db")
col_name = os.getenv("COLLECTION_NAME", "retie_docs")
print("CHROMA_DIR:", persist, "COLLECTION_NAME:", col_name)
c = chromadb.PersistentClient(path=persist)
col = c.get_collection(col_name)
print("COUNT:", col.count())
print("PEEK:", col.peek(1))
