# app/services/mongo_store.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId

_client: Optional[MongoClient] = None
_db = None
_fs: Optional[GridFS] = None

def _get_db_and_fs() -> tuple:
    """Return (db, gridfs) singletons."""
    global _client, _db, _fs
    if _fs is not None:
        return _db, _fs

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not configured")

    db_name = os.getenv("MONGO_DB", "retie")
    bucket = os.getenv("MONGO_BUCKET", "images")

    _client = MongoClient(mongo_uri, connectTimeoutMS=10000, socketTimeoutMS=20000)
    _db = _client[db_name]
    _fs = GridFS(_db, collection=bucket)

    # helpful indexes for queries (safe to call repeatedly)
    _db[f"{bucket}.files"].create_index("uploadDate")
    _db[f"{bucket}.files"].create_index("metadata.chat_id")
    _db[f"{bucket}.files"].create_index("metadata.user_id")
    return _db, _fs

def save_image_from_path(
    path,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store a local file into GridFS. Return the file id as hex string."""
    _, fs = _get_db_and_fs()
    with open(path, "rb") as f:
        file_id = fs.put(
            f,
            filename=filename or os.path.basename(str(path)),
            content_type=content_type or "image/jpeg",
            metadata={**(metadata or {}), "uploaded_at": datetime.utcnow()},
        )
    return str(file_id)

def get_image_bytes(file_id: str) -> Tuple[bytes, str, str]:
    """Fetch file by id. Returns (bytes, content_type, filename)."""
    _, fs = _get_db_and_fs()
    grid_out = fs.get(ObjectId(file_id))
    data = grid_out.read()
    ctype = getattr(grid_out, "content_type", "application/octet-stream")
    fname = getattr(grid_out, "filename", f"{file_id}.bin")
    return data, ctype, fname
