#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List
from app.ingestion.indexer import index_file, SUPPORTED
from app.retriever.chroma_client import get_collection, drop_collection

def main():
    p = argparse.ArgumentParser(description="Indexa documentos en Chroma (PDF) con parser seleccionable")
    p.add_argument("folder", type=str, help="Carpeta con documentos PDF")
    p.add_argument("--collection", default=None, help="Colección destino (por agente)")
    p.add_argument("--parser", default="pdfplumber", help="Parser: pdfplumber | pymupdf | hybrid")
    p.add_argument("--rebuild", action="store_true", help="Limpiar la colección antes de indexar")
    p.add_argument("--file", default=None, help="Indexar un PDF específico dentro de la carpeta")

    args = p.parse_args()

    base = Path(args.folder).resolve()
    assert base.exists() and base.is_dir(), f"No existe carpeta: {base}"

    # Manejo de rebuild sin fallar si la colección no existe
    if args.rebuild:
        from app.config import settings as _s
        coll_name = args.collection or _s.COLLECTION_NAME
        try:
            drop_collection(coll_name)
            print(f"🧹 Colección eliminada: {coll_name}")
        except ValueError:
            print(f"⚠ La colección {coll_name} no existía, se creará automáticamente.")

    # Lista de archivos a indexar
    if args.file:
        f = Path(args.file).resolve()
        assert f.exists() and f.suffix.lower() == ".pdf", f"PDF no existe: {f}"
        files: List[Path] = [f]
    else:
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
