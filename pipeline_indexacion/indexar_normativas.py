"""Indexador de normativas eléctricas colombianas en ChromaDB + MinIO.

Flujo:
  1. Indexa cada PDF localmente en ChromaDB (colección única "normativas";
     la vigencia de cada documento queda en la metadata `vigente`).
  2. Elimina data/chroma_db/ del bucket MinIO (solo si la indexación terminó bien).
  3. Sube la base de datos generada a MinIO.

Uso:
  python pipeline_indexacion/indexar_normativas.py
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
PDF_DIR    = _ROOT / "docs"
CHROMA_DIR = _ROOT / "chroma_local"
# Los embeddings se generan con retie_agent.llm.embedder (text-embedding-3-small,
# 1536 dims): DEBE ser el mismo modelo con el que el agente embebe las consultas.
# El modelo HuggingFace anterior (768 dims) dejaba el índice inutilizable para
# el dense retrieval del agente desplegado.

MINIO_BUCKET  = "embeddings-store"
MINIO_PREFIX  = "data/chroma_db"

# ---------------------------------------------------------------------------
# Registro de documentos
# ---------------------------------------------------------------------------
DOCUMENT_REGISTRY = [
    {
        "doc_id":   "ntc2050_v2",
        "filename": "NTC 2050 V2 Codigo Electrico Colombiano - 2a actualización.pdf",
        "doc_name": "NTC 2050 V2 - Código Eléctrico Colombiano",
        "doc_tipo": "codigo_electrico",
        "vigente":  True,
        "entidad":  "ICONTEC",
        "version":  "2a actualización",
        "alias":    "NTC 2050|código eléctrico|norma eléctrica|documento 1|el actual|la norma actual",
        "collection": "normativas",
    },
    {
        "doc_id":   "ntc2050_erratas",
        "filename": "NTC_2050-Fe-de-erratas.pdf",
        "doc_name": "NTC 2050 - Fe de Erratas",
        "doc_tipo": "errata",
        "vigente":  True,
        "entidad":  "ICONTEC",
        "version":  "fe de erratas V2",
        "alias":    "fe de erratas|erratas|correcciones NTC|documento 2",
        "complementa": "ntc2050_v2",
        "collection": "normativas",
    },
    {
        "doc_id":   "retie_libro1",
        "filename": "2._Libro_1___Disposiciones_Generales.pdf",
        "doc_name": "RETIE - Libro 1: Disposiciones Generales",
        "doc_tipo": "reglamento",
        "vigente":  False,
        "entidad":  "MINMINAS",
        "version":  "histórico",
        "alias":    "libro 1|RETIE libro 1|disposiciones generales|documento 3|libro uno",
        "libro":    1,
        "collection": "normativas",
    },
    {
        "doc_id":   "retie_libro2",
        "filename": "3._Libro_2_-_Productos.pdf",
        "doc_name": "RETIE - Libro 2: Productos",
        "doc_tipo": "reglamento",
        "vigente":  False,
        "entidad":  "MINMINAS",
        "version":  "histórico",
        "alias":    "libro 2|RETIE libro 2|productos eléctricos|documento 4|libro dos",
        "libro":    2,
        "collection": "normativas",
    },
    {
        "doc_id":   "retie_libro3",
        "filename": "4._Libro_3_-_Instalaciones.pdf",
        "doc_name": "RETIE - Libro 3: Instalaciones",
        "doc_tipo": "reglamento",
        "vigente":  False,
        "entidad":  "MINMINAS",
        "version":  "histórico",
        "alias":    "libro 3|RETIE libro 3|instalaciones eléctricas|documento 5|libro tres",
        "libro":    3,
        "collection": "normativas",
    },
    {
        "doc_id":   "retie_libro4",
        "filename": "5._Libro_4_-_Evaluación_de_la_conformidad (1).pdf",
        "doc_name": "RETIE - Libro 4: Evaluación de la Conformidad",
        "doc_tipo": "reglamento",
        "vigente":  False,
        "entidad":  "MINMINAS",
        "version":  "histórico",
        "alias":    "libro 4|RETIE libro 4|evaluación de conformidad|documento 6|libro cuatro",
        "libro":    4,
        "collection": "normativas",
    },
]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
SEPARATORS    = ["\n\n", "\n", ".", " ", ""]


# ---------------------------------------------------------------------------
# Función auxiliar: resolver doc_id desde texto libre
# ---------------------------------------------------------------------------
def resolver_doc_id(texto_usuario: str) -> Optional[str]:
    """Busca el doc_id que mejor coincide con el texto libre del usuario.

    Ejemplo:
        resolver_doc_id("dame el libro 3")  → "retie_libro3"
        resolver_doc_id("la norma actual")  → "ntc2050_v2"
    """
    texto = texto_usuario.lower()
    for doc in DOCUMENT_REGISTRY:
        for alias in doc["alias"].split("|"):
            if alias.strip().lower() in texto:
                return doc["doc_id"]
    return None


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------
def _minio_client():
    from minio import Minio
    # Acepta tanto los nombres del .env raíz como los del .env local del pipeline.
    # Preferimos el endpoint público para correr localmente.
    endpoint = (
        os.getenv("MINIO_ENDPOINT")
        or os.getenv("MINIO_PUBLIC_ENDPOINT")
        or os.getenv("MINIO_PRIVATE_ENDPOINT")
    )
    if not endpoint:
        raise RuntimeError("Falta MINIO_ENDPOINT (o MINIO_PUBLIC_ENDPOINT) en el .env")
    # MINIO_ROOT_USER/PASSWORD son los que usa storage_minio.py (credenciales admin Railway)
    access_key = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Faltan credenciales MinIO (MINIO_ACCESS_KEY / MINIO_ROOT_USER)")

    # Auto-detectar HTTPS: si el endpoint trae scheme explícito úsalo;
    # si no, considerar HTTPS cuando termina en :443 (Railway siempre usa 443+TLS).
    if "://" in endpoint:
        scheme, endpoint = endpoint.split("://", 1)
        secure = scheme == "https"
    else:
        secure = endpoint.endswith(":443")

    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _delete_minio_prefix(client, bucket: str, prefix: str) -> int:
    """Delete all objects under prefix. Returns count deleted."""
    from minio.deleteobjects import DeleteObject
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        return 0
    delete_list = [DeleteObject(o.object_name) for o in objects]
    errors = list(client.remove_objects(bucket, delete_list))
    if errors:
        for err in errors:
            logging.warning("MinIO delete error: %s", err)
    return len(objects) - len(errors)


def _upload_chroma_to_minio(client, bucket: str, prefix: str, local_dir: Path) -> int:
    """Upload every file under local_dir to bucket/prefix/. Returns count uploaded."""
    count = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}"
        client.fput_object(bucket, key, str(path))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _find_pdf(filename: str) -> Optional[Path]:
    """Locate the PDF trying exact name and without extension."""
    candidates = [
        PDF_DIR / filename,
        PDF_DIR / Path(filename).stem,
    ]
    # Also try glob for case/accent variations
    stem = Path(filename).stem
    for p in candidates:
        if p.exists():
            return p
    for p in PDF_DIR.glob(f"{stem}*"):
        return p
    return None


def _build_metadata(doc: dict, page: int) -> dict:
    """Build primitive-only metadata dict for a chunk."""
    meta = {
        "doc_id":   doc["doc_id"],
        "doc_name": doc["doc_name"],
        "doc_tipo": doc["doc_tipo"],
        "vigente":  doc["vigente"],
        "entidad":  doc["entidad"],
        "version":  doc["version"],
        "pagina":   page,
        "alias":    doc["alias"],
    }
    if "libro" in doc:
        meta["libro"] = doc["libro"]
    if "complementa" in doc:
        meta["complementa"] = doc["complementa"]
    return meta


def run_indexing() -> dict[str, int]:
    """Index all documents. Returns {collection_name: chunk_count}."""
    import chromadb
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from retie_agent.llm.embedder import embed_texts
    from retie_agent.config import settings

    logging.info("Embeddings vía OpenAI: %s (mismo modelo que el agente)",
                 settings.EMBEDDING_MODEL)

    # Fresh local DB
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # hnsw:space=cosine para que las distancias coincidan con el umbral del agente
    # (RAG_DISTANCE_THRESHOLD asume distancia coseno, no L2).
    _collection_meta = {"hnsw:space": "cosine"}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    counts: dict[str, int] = {}

    for doc in DOCUMENT_REGISTRY:
        pdf_path = _find_pdf(doc["filename"])
        if pdf_path is None:
            logging.warning("  [SKIP] No encontrado: %s", doc["filename"])
            continue

        logging.info("Procesando: %s", doc["doc_name"])
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()

        chunks = splitter.split_documents(pages)
        logging.info("  %d chunks de %d páginas", len(chunks), len(pages))

        col = chroma.get_or_create_collection(doc["collection"], metadata=_collection_meta)
        collection_name = doc["collection"]

        ids, texts, metas = [], [], []
        for i, chunk in enumerate(chunks):
            page_num = int(chunk.metadata.get("page", 0)) + 1  # 1-based
            meta = _build_metadata(doc, page_num)
            ids.append(f"{doc['doc_id']}_chunk_{i:05d}")
            texts.append(chunk.page_content)
            metas.append(meta)

        # Embed and add in batches (límite OpenAI ~300k tokens por request;
        # 200 chunks de ≤800 chars quedan muy por debajo).
        batch = 200
        for start in range(0, len(texts), batch):
            end = start + batch
            embeddings = embed_texts(texts[start:end])
            col.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings,
                metadatas=metas[start:end],
            )
            if start // batch % 10 == 0:
                logging.info("  … %d/%d chunks embebidos", min(end, len(texts)), len(texts))

        counts[collection_name] = counts.get(collection_name, 0) + len(texts)
        logging.info("  ✅ %s → colección '%s'", doc["doc_id"], collection_name)

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Cargar credenciales
    # Primero el .env raíz del proyecto (tiene las credenciales reales de Railway),
    # luego el .env local de esta carpeta para sobreescribir si es necesario.
    root_env = _ROOT.parent / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
    local_env = _ROOT / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=False)  # no pisa valores ya cargados

    # 2. Conectar a MinIO
    _ep = (
        os.getenv("MINIO_ENDPOINT")
        or os.getenv("MINIO_PUBLIC_ENDPOINT")
        or os.getenv("MINIO_PRIVATE_ENDPOINT")
        or "(no configurado)"
    )
    logging.info("Conectando a MinIO en %s ...", _ep)
    client = _minio_client()

    if not client.bucket_exists(MINIO_BUCKET):
        logging.warning("Bucket '%s' no existe, se creará.", MINIO_BUCKET)
        client.make_bucket(MINIO_BUCKET)

    # 3. Indexar documentos PRIMERO. Solo si termina bien se toca MinIO:
    #    borrar el bucket antes de indexar dejaba el índice remoto vacío si
    #    la indexación fallaba a mitad de camino.
    logging.info("\n── Iniciando indexación ──")
    counts = run_indexing()
    if not counts:
        raise RuntimeError(
            "La indexación no produjo chunks (¿faltan los PDFs en docs/?). "
            "Se conserva el índice existente en MinIO."
        )

    # 4. Resumen local
    logging.info("\n── Resumen de indexación ──")
    for col, n in counts.items():
        logging.info("  %-22s  %d chunks", col, n)

    # 5. Reemplazar el índice en MinIO (borrar + subir)
    logging.info("Eliminando %s/%s/ ...", MINIO_BUCKET, MINIO_PREFIX)
    deleted = _delete_minio_prefix(client, MINIO_BUCKET, MINIO_PREFIX + "/")
    logging.info("  %d objetos eliminados.", deleted)

    logging.info("\nSubiendo ChromaDB a MinIO (%s/%s/) ...", MINIO_BUCKET, MINIO_PREFIX)
    uploaded = _upload_chroma_to_minio(client, MINIO_BUCKET, MINIO_PREFIX, CHROMA_DIR)

    # 8. Confirmación
    logging.info("  ✅ %d archivos subidos a MinIO.", uploaded)
    logging.info("Indexación completa.")


if __name__ == "__main__":
    main()
