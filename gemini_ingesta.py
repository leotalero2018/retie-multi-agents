"""Ingesta one-time: crea un File Search Store de Gemini y sube los PDF del RETIE.

El spike (gemini_client.py) solo CONSULTA un store existente; este script lo crea
y lo llena. Se corre una sola vez. Imprime el nombre del store para el .env.

Uso:
  export GEMINI_API_KEY="tu_key"
  python gemini_ingesta.py /ruta/a/carpeta/con/pdfs
  python gemini_ingesta.py /ruta/a/carpeta --store fileSearchStores/xxx   # reusar store

Requiere:  pip install google-genai
"""
import os
import sys
import time
from pathlib import Path

# Carga el .env del repo si existe (para tomar GEMINI_API_KEY sin exportarla a mano).
try:
    from dotenv import load_dotenv
    _repo_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(_repo_env if _repo_env.exists() else None)
except Exception:
    pass

from google import genai

DISPLAY_NAME = "retie-corpus"
EMBEDDING_MODEL = "models/gemini-embedding-2"


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python gemini_ingesta.py <carpeta_pdfs> [--store <nombre>]")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser()
    if not folder.is_dir():
        print(f"ERROR: no es una carpeta: {folder}")
        sys.exit(1)

    reuse_store = None
    if "--store" in sys.argv:
        reuse_store = sys.argv[sys.argv.index("--store") + 1]

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: exporta GEMINI_API_KEY primero.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # 1. Crear (o reusar) el store
    if reuse_store:
        store_name = reuse_store
        print(f"Reusando store: {store_name}")
    else:
        store = client.file_search_stores.create(
            config={"display_name": DISPLAY_NAME, "embedding_model": EMBEDDING_MODEL}
        )
        store_name = store.name
        print(f"Store creado: {store_name}")

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: no hay PDFs en {folder}")
        sys.exit(1)
    print(f"PDFs a subir: {len(pdfs)}\n")

    # 2. Subir cada PDF y esperar la indexación
    ok, err = 0, 0
    for pdf in pdfs:
        print(f"  Subiendo {pdf.name} ...", end=" ", flush=True)
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(pdf),
                file_search_store_name=store_name,
                config={"display_name": pdf.stem},
            )
            while not op.done:
                time.sleep(5)
                op = client.operations.get(op)
            print("OK")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            err += 1

    print(f"\nListo. Subidos: {ok}  Errores: {err}")
    print("\n=== Agrega esto a tu .env ===")
    print(f'GEMINI_FILE_SEARCH_STORE="{store_name}"')


if __name__ == "__main__":
    main()
