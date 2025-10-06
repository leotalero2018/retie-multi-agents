# app/services/mongo_store.py
from __future__ import annotations

import os
import mimetypes
import logging
from typing import Optional, Dict, Any, Tuple, List

from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from gridfs import GridFS

log = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db = None
_fs: Optional[GridFS] = None
_bucket_name: str = os.getenv("MONGO_BUCKET", "images")


def _get_db_and_fs():
    """Return (db, gridfs) singletons."""
    global _client, _db, _fs, _bucket_name

    if _fs is not None:
        return _db, _fs

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not configured")

    db_name = os.getenv("MONGO_DB", "retie")
    _bucket_name = os.getenv("MONGO_BUCKET", "images")

    _client = MongoClient(mongo_uri, connectTimeoutMS=10000, socketTimeoutMS=20000)
    _db = _client[db_name]
    _fs = GridFS(_db, collection=_bucket_name)

    # Helpful indexes for lookups; safe to call repeatedly
    _db[f"{_bucket_name}.files"].create_index("uploadDate")
    _db[f"{_bucket_name}.files"].create_index("metadata.chat_id")
    _db[f"{_bucket_name}.files"].create_index("metadata.user_id")
    return _db, _fs


def save_image_from_path(
    path,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Store a local file into GridFS. Returns the file id as hex string.
    """
    db, fs = _get_db_and_fs()
    filename = filename or os.path.basename(str(path))
    if not content_type:
        content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"

    meta = {**(metadata or {})}
    meta.setdefault("content_type", content_type)
    meta.setdefault("uploaded_at", datetime.utcnow())

    with open(path, "rb") as f:
        file_id = fs.put(
            f,
            filename=filename,
            content_type=content_type,
            metadata=meta,
        )
    oid = str(file_id)
    log.info("[IMG] Saved id=%s filename=%s meta=%s", oid, filename, meta)
    return oid


def get_image_bytes(file_id: str) -> Tuple[bytes, str, str]:
    """
    Fetch file by id. Returns (bytes, content_type, filename).
    """
    db, fs = _get_db_and_fs()
    go = fs.get(ObjectId(file_id))
    data = go.read()
    ctype = getattr(go, "content_type", "application/octet-stream")
    fname = getattr(go, "filename", f"{file_id}.bin")
    return data, ctype, fname


def list_recent_files(limit: int = 20) -> List[Dict[str, Any]]:
    """
    List recent GridFS files with basic metadata.
    Returns: [{id, filename, bytes, uploaded, metadata}, ...]
    """
    db, _ = _get_db_and_fs()
    coll = db[f"{_bucket_name}.files"]
    cur = coll.find(
        {}, {"_id": 1, "filename": 1, "length": 1, "uploadDate": 1, "metadata": 1}
    ).sort("uploadDate", -1).limit(max(1, min(limit, 200)))

    out: List[Dict[str, Any]] = []
    for d in cur:
        out.append({
            "id": str(d["_id"]),
            "filename": d.get("filename"),
            "bytes": d.get("length"),
            "uploaded": d.get("uploadDate"),
            "metadata": d.get("metadata", {}),
        })
    return out
