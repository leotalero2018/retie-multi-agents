# app/api/ingest.py
# Triggers a background sync of the Chroma DB from the Railway bucket,
# or rebuilds the vector DB from local PDFs and (optionally) uploads it.

import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import Optional

from app.ingestion.pipeline import index_folder
from app.services.storage_s3 import download_folder, upload_folder

router = APIRouter(prefix="/ingest", tags=["ingest"])

# ---- Config helpers ----
def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise HTTPException(status_code=500, detail=f"{name} not configured")
    return val

def _check_secret(secret: str = Query(..., description="INGEST_SECRET")) -> None:
    expected = os.getenv("INGEST_SECRET")
    if not expected:
        raise HTTPException(status_code=500, detail="INGEST_SECRET not configured")
    if secret != expected:
        raise HTTPException(status_code=403, detail="Invalid secret")

# ---- Background tasks ----
def _task_sync_from_bucket(prefix: str):
    bucket = os.getenv("S3_BUCKET_NAME")
    persist = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    download_folder(bucket=bucket, prefix=prefix, local_dir=persist)

def _task_rebuild_and_optionally_upload(source: str, persist: str, engine: str, upload: bool, prefix: str):
    # Re-index from local PDFs
    index_folder(source_dir=source, persist_dir=persist, engine=engine)
    # Optionally upload the new DB to S3
    if upload:
        bucket = os.getenv("S3_BUCKET_NAME")
        upload_folder(local_dir=persist, bucket=bucket, prefix=prefix)

# ---- Endpoints ----
@router.post("/sync")
def sync_from_bucket(
    background_tasks: BackgroundTasks,
    secret: str = Query(...),
    prefix: str = Query(default="chroma_db/"),
):
    """
    Download the persisted Chroma folder from S3 to CHROMA_PERSIST_DIR.
    Use when you already uploaded the vector DB and just want the app to use it.
    """
    _check_secret(secret)
    _require_env("S3_BUCKET_NAME")
    background_tasks.add_task(_task_sync_from_bucket, prefix)
    return {"ok": True, "message": "Sync from bucket started", "prefix": prefix}

@router.post("/rebuild")
def rebuild_from_docs(
    background_tasks: BackgroundTasks,
    secret: str = Query(...),
    source: str = Query(default="./docs"),
    engine: str = Query(default="pymupdf", regex="^(pymupdf|pdfplumber)$"),
    upload: bool = Query(default=False),
    prefix: str = Query(default="chroma_db/"),
):
    """
    Rebuild the Chroma DB from local PDFs (source folder) and write to CHROMA_PERSIST_DIR.
    Optional: upload the rebuilt DB to S3 (upload=true).
    """
    _check_secret(secret)
    persist = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    _require_env("S3_BUCKET_NAME") if upload else None

    background_tasks.add_task(
        _task_rebuild_and_optionally_upload,
        source, persist, engine, upload, prefix
    )
    return {
        "ok": True,
        "message": "Rebuild started",
        "source": source,
        "engine": engine,
        "upload": upload,
        "prefix": prefix,
        "persist_dir": persist,
    }
