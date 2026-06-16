"""
Script de setup inicial para NotebookLM MCP.

Uso típico (ejecutar en LOCAL, no en Railway):

  # Terminal 1 — arrancar el servidor MCP:
  set PLAYWRIGHT_USER_DATA_DIR=./nlm-session
  npx notebooklm-mcp@latest --transport http --port 3000

  # Terminal 2 — autenticar y configurar:
  python setup_notebooklm.py           # login + configuración
  python setup_notebooklm.py --upload  # sube la sesión a MinIO (para Railway)
  python setup_notebooklm.py --pack    # solo comprime a ./sessions (subes a mano)

Flags opcionales: --session-dir <ruta> (browser_state de origen),
                  --out-dir <ruta>     (carpeta de salida de --pack, def. ./sessions)

La sesión de Playwright (browser_state) se empaqueta como nlm-browser-state.tar.gz.
--upload la sube a MinIO; --pack la deja local para subirla manualmente al bucket
con esa misma clave. Railway la descarga al arrancar para restaurar la sesión.
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

    endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT", "(no definido)")
    print(f"Endpoint MinIO (público): {endpoint}")
    # Falla rápido si el endpoint no responde, en vez de colgarse 5 min.
    os.environ.setdefault("MINIO_CONNECT_TIMEOUT", "10")
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
        low = str(e).lower()
        if "timed out" in low or "timeout" in low or "max retries" in low or "connection" in low:
            print(
                "\n  El endpoint público no respondió (problema de alcanzabilidad, no del archivo).\n"
                f"  Endpoint usado: MINIO_PUBLIC_ENDPOINT={endpoint}\n\n"
                "  Verifica que MINIO_PUBLIC_ENDPOINT apunte al host:puerto correctos:\n"
                "    • En Railway, el acceso público a MinIO suele ser por un TCP Proxy con un\n"
                "      puerto propio (NO el 443). Revísalo en Settings → Networking → TCP Proxy\n"
                "      del servicio MinIO y usa, p. ej.:\n"
                "        MINIO_PUBLIC_ENDPOINT=https://bucket-production-xxxx.up.railway.app:<PUERTO>\n"
                "    • Comprueba la conectividad: curl -v "
                "https://<host>:<puerto>/minio/health/live\n"
                "  Alternativa: ejecuta este --upload desde un entorno con acceso a la red\n"
                "  interna de Railway (MINIO_PRIVATE_ENDPOINT)."
            )
        sys.exit(1)
    finally:
        if _priv:
            os.environ["MINIO_PRIVATE_ENDPOINT"] = _priv


def _pack_session(session_dir: str | None = None, out_dir: str | None = None) -> None:
    """Comprime la sesión a un .tar.gz LOCAL (carpeta ./sessions) para subirla a mano.

    No toca MinIO: útil cuando el bucket no es alcanzable. Imprime la ruta del
    archivo y el bucket/clave exactos donde debe subirse para que Railway lo use.
    """
    from dotenv import load_dotenv
    load_dotenv()
    from retie_agent.services.nlm_session import (
        pack_nlm_session,
        default_browser_state_dir,
        _MINIO_KEY,
    )

    src = Path(session_dir) if session_dir else default_browser_state_dir()
    print(f"\nDirectorio de sesión a comprimir: {src}")
    if not src.exists():
        print(f"\n  ERROR: No se encontró el directorio de sesión: {src}")
        print("  Autentica primero con: python setup_notebooklm.py")
        print("  O pasa la ruta explícita: python setup_notebooklm.py --pack --session-dir <ruta>")
        sys.exit(1)

    print("Comprimiendo a archivo local ...")
    try:
        out_path = pack_nlm_session(str(src), out_dir)
    except Exception as e:
        print(f"  ERROR al comprimir: {e}")
        sys.exit(1)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n  ✅ Sesión comprimida: {out_path}  ({size_kb:.1f} KB)")

    # Bucket/clave destino (solo lee variables de entorno, sin red).
    bucket = "<tu_bucket_MinIO>"
    try:
        from retie_agent.services.nlm_session import _bucket
        bucket = _bucket()
    except Exception:
        pass
    print("\n  Súbelo manualmente al bucket de MinIO con ESTA clave exacta:")
    print(f"    bucket : {bucket}")
    print(f"    key    : {_MINIO_KEY}   (en la raíz del bucket, sin prefijo)")
    print("\n  Railway descargará esa clave al arrancar para restaurar la sesión de NotebookLM.")


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
        print(f"  Subir a MinIO (Railway):   python setup_notebooklm.py --upload")
        print(f"  O comprimir para subir a mano: python setup_notebooklm.py --pack")
    print("  Puedes iniciar el bot.")


def _flag_value(name: str) -> str | None:
    """Devuelve el valor que sigue a un flag (p. ej. --session-dir <ruta>)."""
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


if __name__ == "__main__":
    if "--pack" in sys.argv:
        # Comprime a ./sessions (o --out-dir) para subir a mano; no usa MinIO.
        _pack_session(_flag_value("--session-dir"), _flag_value("--out-dir"))
    elif "--upload" in sys.argv:
        _upload_session(_flag_value("--session-dir"))
    else:
        main()
