# app/services/whisper.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import httpx

from app.config import settings


def _build_openai_client():
    """Create an OpenAI client honoring proxies and custom base URL (Azure/OpenRouter/etc.)."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or settings.")

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
    http_client = httpx.Client(proxies=proxy, timeout=120) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None

    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def transcribe_audio(filepath: Path, language: Optional[str] = "es") -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper.
    - File: local path to wav/mp3/ogg/opus/m4a/flac/etc.
    - Language: optional hint ('es' by default).
    Returns plain text (str).
    """
    provider = (getattr(settings, "WHISPER_PROVIDER", "openai") or "openai").lower()
    if provider != "openai":
        raise NotImplementedError("Only WHISPER_PROVIDER='openai' is implemented right now.")

    # Cheaper & very reliable:
    model = getattr(settings, "WHISPER_MODEL", None) or "whisper-1"

    client = _build_openai_client()

    p = Path(filepath)
    if not p.exists():
        raise RuntimeError(f"Audio file not found: {filepath}")

    try:
        with open(p, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,          # "whisper-1" (recommended) or "gpt-4o-transcribe"
                file=f,
                language=language,    # hint
                response_format="text"
            )
        return getattr(resp, "text", str(resp))  # SDK returns .text for response_format="text"
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from None
