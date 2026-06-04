#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import defaultdict

# Agregar carpeta raíz al path
root = Path(__file__).parent.parent.resolve()
sys.path.append(str(root))

from retie_agent.retriever.chroma_client import get_collection

collections = ["retie_plumber", "retie_pymupdf"]

for coll_name in collections:
    print(f"\n--- Colección: {coll_name} ---")
    try:
        coll = get_collection(coll_name)
        metadatas = coll.get()["metadatas"]

        # Contar chunks por PDF
        chunks_por_pdf = defaultdict(int)
        for m in metadatas:
            chunks_por_pdf[m["source"]] += 1

        # Mostrar solo PDFs y cantidad de chunks
        for pdf, count in chunks_por_pdf.items():
            print(f"{pdf}: {count} chunks")

    except Exception as e:
        print(f"Error al revisar colección {coll_name}: {e}")
