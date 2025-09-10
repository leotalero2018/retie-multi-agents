#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List
from app.ingestion.indexer import index_file, SUPPORTED
from app.retriever.chroma_client import get_collection
from app.retriever.chroma_client import get_collection, drop_collection

def main():
    p = argparse.ArgumentParser(description="Indexa documentos en Chroma (PDF) con parser seleccionable")
    p.add_argument("folder", type=str, help="Carpeta con documentos PDF")
    p.add_argument("--collection", default=None, help="Colección destino (por agente)")
    p.add_argument("--parser", default="pdfplumber", help="Parser: pdfplumber | pymupdf | hybrid")
    p.add_argument("--rebuild", action="store_true", help="Limpiar la colección antes de indexar")
    args = p.parse_args()

    base = Path(args.folder).resolve()
    assert base.exists() and base.is_dir(), f"No existe carpeta: {base}"

    if args.rebuild:
        target = args.collection  # puede ser None -> usaremos el default más abajo
        name = target or "DEFAULT"
        # Si no pasaste --collection, usa la de settings
        from app.config import settings as _s
        coll_name = target or _s.COLLECTION_NAME

        drop_collection(coll_name)   # <-- elimina completamente
        print(f"🧹 Colección eliminada: {coll_name}")
        # Se recreará automáticamente al primer get_or_create (get_collection)

    files: List[Path] = [p for p in base.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"]
    if not files:
        print("No se encontraron PDFs.")
        return

    print(f"Indexando {len(files)} PDF(s) en colección '{args.collection or 'DEFAULT'}' con parser '{args.parser}'...")
    total_chunks = 0
    for f in files:
        info = index_file(f, collection_name=args.collection, parser_name=args.parser)
        print(f"✔ {info['file']} -> {info['chunks']} chunks")
        total_chunks += info["chunks"]
    print(f"\n✅ Terminado. Total chunks: {total_chunks}")

if __name__ == "__main__":
    main()