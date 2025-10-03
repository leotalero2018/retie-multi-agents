# app/api/debug.py
# Debug endpoint to list objects in your Railway S3-compatible bucket (e.g., MinIO/Spaces/S3).
# Requires env: S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY
# Optional protection: MINIO_DEBUG_SECRET (query param ?secret=)

import os
from fastapi import APIRouter, HTTPException, Query
from typing import List
import boto3

router = APIRouter()

def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION", "auto"),
    )

@router.get("/debug/bucket")
def debug_bucket(secret: str | None = Query(default=None), prefix: str = "chroma_db/"):
    """
    Lists objects under the given prefix in S3 bucket (Railway).
    Protected by MINIO_DEBUG_SECRET if set.
    """
    debug_secret = os.getenv("MINIO_DEBUG_SECRET")
    if debug_secret and secret != debug_secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME not configured")

    try:
        s3 = _s3()
        keys: List[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return {"ok": True, "bucket": bucket, "prefix": prefix, "objects": keys}
    except Exception as e:
        return {"ok": False, "error": str(e)}
