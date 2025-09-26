# app/api/ingest.py
import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.ingestion.index_minio import ingest_from_minio

router = APIRouter(prefix="/ingest", tags=["ingest"])

def verify_ingest_secret(x_secret: str = Depends(lambda: os.getenv("INGEST_SECRET"))):
    if not x_secret:
        raise HTTPException(status_code=500, detail="INGEST_SECRET not configured")
    return x_secret

@router.post("/minio")
def ingest_minio(background_tasks: BackgroundTasks, secret: str, rebuild: bool = False, _: str = Depends(verify_ingest_secret)):
    """
    Lanza la ingesta de PDFs desde MinIO en background.
    Debes pasar ?secret=INGEST_SECRET en la query.
    - rebuild=True fuerza reindexación.
    """
    if secret != os.getenv("INGEST_SECRET"):
        raise HTTPException(status_code=403, detail="Invalid secret")

    background_tasks.add_task(ingest_from_minio, rebuild=rebuild)
    return {"ok": True, "message": "Ingest started in background"}
