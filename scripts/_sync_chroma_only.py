"""Descarga solo los archivos core de ChromaDB desde MinIO (sin assets/figures)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from minio import Minio

endpoint_raw = os.getenv("MINIO_PUBLIC_ENDPOINT", "")
endpoint = endpoint_raw.removeprefix("https://").removeprefix("http://")
secure = endpoint_raw.startswith("https://")

access = os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_ACCESS_KEY", "")
secret = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_SECRET_KEY", "")
bucket = os.getenv("MINIO_BUCKET_NAME", "embeddings-store")
prefix = os.getenv("MINIO_PREFIX", "chroma_v2/").rstrip("/") + "/"
local_dir = Path(os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DB_DIR") or "data/chroma_db")

print(f"Endpoint : {endpoint}  secure={secure}")
print(f"Bucket   : {bucket}")
print(f"Prefijo  : {prefix}")
print(f"Destino  : {local_dir}")
print()

client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)

objs = list(client.list_objects(bucket, prefix=prefix, recursive=True))
core = [o for o in objs if "/assets/" not in o.object_name]
assets = [o for o in objs if "/assets/" in o.object_name]

print(f"Total objetos  : {len(objs)}")
print(f"  Core (DB)    : {len(core)}")
print(f"  Assets       : {len(assets)}  (omitidos para ahorrar espacio)")
print()

local_dir.mkdir(parents=True, exist_ok=True)

downloaded = 0
errors = 0
for obj in core:
    rel = obj.object_name[len(prefix):]
    dest = local_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    size_kb = (getattr(obj, "size", 0) or 0) / 1024
    try:
        client.fget_object(bucket, obj.object_name, str(dest))
        print(f"  OK  {rel}  ({size_kb:.0f} KB)")
        downloaded += 1
    except Exception as e:
        print(f"  ERR {rel}: {e}")
        errors += 1

print()
print(f"Descargados: {downloaded}  Errores: {errors}")
if errors == 0:
    print(f"ChromaDB core sincronizado en: {local_dir.resolve()}")
else:
    print("Hubo errores, revisa los mensajes anteriores.")
    sys.exit(1)
