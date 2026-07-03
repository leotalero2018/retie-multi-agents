import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from retie_agent.agent.notebooklm_client import NotebookLMClient

client = NotebookLMClient(base_url="http://localhost:3000", timeout=60.0)
print("Session ID cargado:", client._mcp_session)

print("\n[1] Verificando autenticacion con Google...")
auth = client.is_authenticated()
print("   authenticated =", auth)

print("\n[2] Listando notebooks...")
try:
    resp = client._call_tool("list_notebooks", {})
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "{}") if isinstance(content, list) and content else "{}"
    data = json.loads(text) if isinstance(text, str) else text
    if isinstance(data, dict):
        inner = data.get("data", data)
        notebooks = inner.get("notebooks", []) if isinstance(inner, dict) else (inner if isinstance(inner, list) else [])
    else:
        notebooks = []
    print(f"   {len(notebooks)} notebook(s):")
    for nb in notebooks:
        print(f"   - [{nb.get('id')}] {nb.get('name') or nb.get('title','?')}")
    if not notebooks:
        print("   (ninguno — agrega el notebook de RETIE)")
except Exception as e:
    print("   Error:", e)

print("\n[3] Agregar notebook de RETIE (pega el share link o Enter para saltar):")
url = input("   URL: ").strip()
if url:
    titulo = input("   Titulo [RETIE]: ").strip() or "RETIE"
    try:
        resp = client._call_tool("add_notebook", {"url": url, "title": titulo})
        content = resp.get("result", {}).get("content", [])
        text = content[0].get("text", "") if isinstance(content, list) and content else ""
        data = json.loads(text) if text else {}
        nb_id = data.get("id") or data.get("data", {}).get("id", "")
        print(f"\n   Notebook agregado. ID: {nb_id}")
        print(f"\n   Agrega a tu .env:")
        print(f"   NOTEBOOKLM_NOTEBOOK_ID={nb_id}")
    except Exception as e:
        print("   Error:", e)

print("\nListo. Puedes subir la sesion a MinIO con:")
print("  python setup_notebooklm.py --upload")
