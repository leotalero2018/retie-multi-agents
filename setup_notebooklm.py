"""
Script de setup inicial para NotebookLM MCP.

Uso típico (ejecutar en LOCAL, no en Railway):

  # Terminal 1 — arrancar el servidor MCP:
  set PLAYWRIGHT_USER_DATA_DIR=./nlm-session
  npx notebooklm-mcp@latest --transport http --port 3000

  # Terminal 2 — autenticar y configurar:
  python setup_notebooklm.py           # login + configuración
  python setup_notebooklm.py --upload  # sube la sesión a MinIO (para Railway)

La sesión de Playwright se guarda en ./nlm-session y se sube a MinIO como
nlm-session.tar.gz para que Railway la descargue automáticamente al arrancar.
"""
import json
import os
import sys
from pathlib import Path
from retie_agent.agent.notebooklm_client import NotebookLMClient, NotebookLMError

NLM_URL = "http://localhost:3000"
DEFAULT_SESSION_DIR = str(Path(__file__).parent / "nlm-session")


def _upload_session(session_dir: str | None = None) -> None:
    """Sube el browser_state de Playwright a MinIO para que Railway lo descargue al arrancar."""
    from dotenv import load_dotenv
    load_dotenv()
    from retie_agent.services.nlm_session import upload_nlm_session, default_browser_state_dir

    # Determinar la ruta fuente
    if session_dir:
        src = Path(session_dir)
    else:
        src = default_browser_state_dir()

    print(f"\nDirectorio de sesión a subir: {src}")
    if not src.exists():
        print(f"\n  ERROR: No se encontró el directorio de sesión: {src}")
        print("  Asegúrate de haber autenticado primero con: python setup_notebooklm.py")
        print(f"  O pasa la ruta explícita: python setup_notebooklm.py --upload --session-dir <ruta>")
        sys.exit(1)

    # MINIO_PRIVATE_ENDPOINT solo es accesible desde dentro de Railway.
    # Al correr localmente lo suprimimos para que _client() use MINIO_PUBLIC_ENDPOINT.
    _priv = os.environ.pop("MINIO_PRIVATE_ENDPOINT", None)
    print("Comprimiendo y subiendo a MinIO ...")
    try:
        upload_nlm_session(str(src))
        print("  ✅ Sesión subida correctamente. Railway la descargará al arrancar.")
        print("\n  Variables necesarias en Railway:")
        print("    NOTEBOOKLM_ENABLED=true")
        print("    NOTEBOOKLM_URL=http://localhost:3000")
        print("    NOTEBOOKLM_NOTEBOOK_ID=<tu_notebook_id>")
    except Exception as e:
        print(f"  ERROR al subir: {e}")
        sys.exit(1)
    finally:
        if _priv:
            os.environ["MINIO_PRIVATE_ENDPOINT"] = _priv


def main() -> None:
    client = NotebookLMClient(base_url=NLM_URL, timeout=60.0)

    print(f"\n[1/4] Conectando al servidor MCP en {NLM_URL} ...")
    try:
        ok = client.initialize()
    except NotebookLMError as e:
        print(f"\n  ERROR: {e}")
        print("\n  Asegúrate de que el servidor esté corriendo:")
        print("  > npx notebooklm-mcp@latest --transport http --port 3000\n")
        sys.exit(1)

    print(f"      MCP session ID: {client._mcp_session}")

    print("\n[2/4] Verificando autenticación con Google ...")
    authenticated = client.is_authenticated()
    print(f"      authenticated = {authenticated}")

    if not authenticated:
        print("\n  El servidor abrirá Chrome para que inicies sesión con tu cuenta Google.")
        print("  Llama a setup_auth ...")
        try:
            resp = client._call_tool("setup_auth", {})
            print(f"      setup_auth response: {json.dumps(resp, indent=2)[:400]}")
        except NotebookLMError as e:
            print(f"      setup_auth error: {e}")
        print("\n  Después de autenticarte en el navegador, vuelve a ejecutar este script.")
        sys.exit(0)

    print("\n[3/4] Listando notebooks en la librería local ...")
    try:
        resp = client._call_tool("list_notebooks", {})
        content = resp.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "{}") if isinstance(content, list) else "{}"
            data = json.loads(text) if isinstance(text, str) else text
            notebooks = data.get("notebooks", [])
            if notebooks:
                print(f"      {len(notebooks)} notebook(s) encontrado(s):")
                for nb in notebooks:
                    print(f"        - [{nb.get('id')}] {nb.get('title', 'Sin título')}")
            else:
                print("      Sin notebooks en la librería local.")
        else:
            print(f"      Respuesta raw: {resp}")
    except Exception as e:
        print(f"      Error listando notebooks: {e}")

    print("\n[4/4] Agregar notebook (opcional) ...")
    share_url = input(
        "\n  Pega el share link de tu notebook en NotebookLM\n"
        "  (Enter para omitir si ya tienes uno en la lista): "
    ).strip()

    if share_url:
        title = input("  Título corto para identificarlo: ").strip() or "RETIE"
        try:
            resp = client._call_tool("add_notebook", {"url": share_url, "title": title})
            content = resp.get("result", {}).get("content", [])
            text = content[0].get("text", "") if isinstance(content, list) and content else ""
            data = json.loads(text) if text else {}
            nb_id = data.get("id", "")
            print(f"\n  Notebook agregado. ID: {nb_id}")
            print(f"\n  Agrega esto a tu .env:")
            print(f"  NOTEBOOKLM_NOTEBOOK_ID={nb_id}")
        except Exception as e:
            print(f"      Error al agregar notebook: {e}")

    print("\n  Setup completo.")
    session_dir = os.environ.get("PLAYWRIGHT_USER_DATA_DIR", DEFAULT_SESSION_DIR)
    if Path(session_dir).exists():
        print(f"\n  Sesión guardada en: {session_dir}")
        print(f"  Para subir a MinIO (Railway): python setup_notebooklm.py --upload")
    print("  Puedes iniciar el bot.")


if __name__ == "__main__":
    if "--upload" in sys.argv:
        # Soporte para --session-dir <ruta> explícita
        session_dir: str | None = None
        if "--session-dir" in sys.argv:
            idx = sys.argv.index("--session-dir")
            if idx + 1 < len(sys.argv):
                session_dir = sys.argv[idx + 1]
        _upload_session(session_dir)
    else:
        main()
