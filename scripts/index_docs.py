#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

from pipeline_indexacion.ingestion.pipeline import index_folder
from pipeline_indexacion.ingestion.indexer import index_file

def main():
    p = argparse.ArgumentParser(description="Indexa PDFs en Chroma con parser seleccionable.")
    p.add_argument("--source", default="./docs", help="Carpeta con PDFs")
    p.add_argument("--out", default=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
                   help="Directorio de persistencia de Chroma")
    p.add_argument("--engine", default="pymupdf", choices=["pymupdf", "pdfplumber"],
                   help="Parser a usar")
    p.add_argument("--file", default=None, help="Indexar un PDF específico (ruta al archivo)")
    p.add_argument("--rebuild", action="store_true", help="(Obsoleto) El pipeline ya borra por 'source'")
    p.add_argument("--upload", action="store_true", help="Subir la DB al bucket S3 (Railway/MinIO/Spaces)")
    p.add_argument("--s3-prefix", default="chroma_db/", help="Prefijo en el bucket S3 para la DB")
    args = p.parse_args()

    persist_dir = args.out
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    if args.file:
        f = Path(args.file).resolve()
        assert f.exists() and f.suffix.lower() == ".pdf", f"PDF no existe: {f}"
        res = index_file(f, parser_name=args.engine)
        print(f"✔ {res['file']} -> {res['chunks']} chunks")
        total = res["chunks"]
    else:
        total = index_folder(args.source, persist_dir, engine=args.engine)
        print(f"✔ Carpeta {args.source} -> {total} chunks")

    if args.upload:
        # Importar solo cuando se usa --upload
        try:
            from retie_agent.services.storage_s3 import upload_folder
        except ModuleNotFoundError:
            raise SystemExit(
                "Falta app/services/storage_s3.py. Crea ese archivo o ejecuta sin --upload."
            )
        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise SystemExit("S3_BUCKET_NAME no está configurado")
        upload_folder(local_dir=persist_dir, bucket=bucket, prefix=args.s3_prefix)
        print(f"⬆️  Subido a s3://{bucket}/{args.s3_prefix}")

    print(f"\n✅ Terminado. Total chunks: {total}")

if __name__ == "__main__":
    main()
