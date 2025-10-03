# app/utils/storage.py
# S3-compatible helpers (Railway bucket, MinIO, DO Spaces, etc.)
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import os
import boto3


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION", "auto"),
    )


def ensure_bucket(bucket: str) -> None:
    """
    Creates the bucket if it doesn't exist (best-effort; safe to call always).
    Not all S3-compatible services allow create_bucket without region; we try and ignore failures.
    """
    s3 = _client()
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except Exception:
        pass
    try:
        # Some providers require LocationConstraint; if it fails, we ignore.
        region = os.getenv("S3_REGION") or "us-east-1"
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region} if region != "us-east-1" else {},
        )
    except Exception:
        # Bucket may already exist globally or provider disallows creation — ignore.
        return


def put_file(bucket: str, local_path: Path, object_name: Optional[str] = None, expires_seconds: int = 7 * 24 * 3600) -> str:
    """
    Uploads a file and returns a pre-signed GET URL valid for `expires_seconds`.
    """
    ensure_bucket(bucket)
    s3 = _client()
    key = object_name or local_path.name
    s3.upload_file(str(local_path), bucket, key)

    # Pre-signed URL
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )
    return url


def build_object_name(chat_id: int, prefix: str, suffix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}/{chat_id}/{ts}.{suffix}"
