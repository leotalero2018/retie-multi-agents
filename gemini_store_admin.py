"""Administra los File Search Stores de Gemini: listar, ver documentos y borrar.

Usa la GEMINI_API_KEY del .env por defecto. Para operar sobre la cuenta retie,
antepón la key en el comando:
    GEMINI_API_KEY="AQ...retie" python gemini_store_admin.py list

Comandos:
    list                          Lista todos los stores de la cuenta.
    docs   <store>                Lista los documentos de un store.
    delete <store>                Borra un store COMPLETO (con todos sus documentos).
    deldoc <store> <documento>    Borra un solo documento de un store.

Ejemplos:
    python gemini_store_admin.py list
    python gemini_store_admin.py docs fileSearchStores/retiecorpus-xxxx
    python gemini_store_admin.py delete fileSearchStores/retiecorpus-xxxx
"""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env if _env.exists() else None)
except Exception:
    pass

from google import genai


def _client():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("ERROR: falta GEMINI_API_KEY (en el .env o antepuesta al comando).")
        sys.exit(1)
    return genai.Client(api_key=key)


def cmd_list(client):
    stores = list(client.file_search_stores.list())
    if not stores:
        print("No hay stores en esta cuenta.")
        return
    print(f"{len(stores)} store(s):\n")
    for s in stores:
        name = getattr(s, "name", "?")
        disp = getattr(s, "display_name", "") or ""
        print(f"  {name}   {disp}")


def cmd_docs(client, store):
    docs = list(client.file_search_stores.documents.list(parent=store))
    if not docs:
        print("Ese store no tiene documentos (o no existe).")
        return
    print(f"{len(docs)} documento(s) en {store}:\n")
    for d in docs:
        name = getattr(d, "name", "?")
        disp = getattr(d, "display_name", "") or ""
        print(f"  {name}   {disp}")


def cmd_delete(client, store):
    # force=True borra el store aunque tenga documentos dentro.
    client.file_search_stores.delete(name=store, config={"force": True})
    print(f"Store borrado: {store}")


def cmd_deldoc(client, store, doc):
    client.file_search_stores.documents.delete(name=doc)
    print(f"Documento borrado: {doc}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    client = _client()
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list(client)
    elif cmd == "docs" and len(sys.argv) >= 3:
        cmd_docs(client, sys.argv[2])
    elif cmd == "delete" and len(sys.argv) >= 3:
        cmd_delete(client, sys.argv[2])
    elif cmd == "deldoc" and len(sys.argv) >= 4:
        cmd_deldoc(client, sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
