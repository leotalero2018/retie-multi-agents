# app/services/history.py
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from retie_agent.services.mongo_store import _get_db_and_fs

log = logging.getLogger(__name__)

_index_ready = False


def _get_collection():
    """Devuelve la colección de historial, creando el índice solo una vez por proceso."""
    global _index_ready
    db, _ = _get_db_and_fs()
    coll = db["chat_history"]
    if not _index_ready:
        try:
            coll.create_index([("session_id", 1), ("created_at", 1)])
            _index_ready = True
        except Exception as e:
            log.warning("Could not ensure chat_history index: %s", e)
    return coll


def get_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Retrieve the last N messages for a given session.
    Returns a list of messages in OpenAI format: [{"role": "user", "content": "..."}, ...]
    """
    try:
        coll = _get_collection()
        cursor = coll.find(
            {"session_id": session_id},
            {"role": 1, "content": 1, "_id": 0}
        ).sort("created_at", -1).limit(limit)
        
        # Reverse to get chronological order
        history = list(cursor)
        history.reverse()
        return history
    except Exception as e:
        log.error("Error retrieving history for session %s: %s", session_id, e)
        return []

def add_message(session_id: str, user_id: str, role: str, content: str):
    """
    Save a new message to the chat history.
    """
    if not content:
        return

    try:
        coll = _get_collection()
        coll.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow()
        })
    except Exception as e:
        log.error("Error saving message to history: %s", e)


def clear_history(session_id: str) -> int:
    """Borra el historial de una sesión. Devuelve cuántos mensajes se eliminaron."""
    try:
        coll = _get_collection()
        result = coll.delete_many({"session_id": session_id})
        return int(result.deleted_count)
    except Exception as e:
        log.error("Error clearing history for session %s: %s", session_id, e)
        return 0
