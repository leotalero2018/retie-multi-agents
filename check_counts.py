from app.retriever.retrieve import search
hits = search("¿Qué es un accidente?", top_k=8, collection_name="retie_docs", distance_threshold=1.0)
print("HITS:", len(hits))
for i,h in enumerate(hits,1):
    m=h["meta"]; print(f"{i}. dist={h['score']:.3f} src={m.get('source')} p={m.get('page')}")
    print("   ", (h["text"][:220].replace("\n"," ") + "..."))