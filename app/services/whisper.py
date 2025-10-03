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

    # Prefer env var directly; fall back to settings
    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or settings.")

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
    http_client = httpx.Client(proxies=proxy, timeout=120) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None

    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def transcribe_audio(filepath: Path, language: Optional[str] = "es") -> str:
    """
    Transcribe an audio file to text.
    - Provider: 'openai' (default via settings.WHISPER_PROVIDER)
    - File: local path to wav/mp3/ogg/opus/m4a/flac/etc.
    - Language: optional hint (e.g., 'es', 'en', 'es+en' isn’t supported by API; pass best guess)
    Returns plain text (str). Raises RuntimeError with a readable message on failure.
    """
    provider = (getattr(settings, "WHISPER_PROVIDER", "openai") or "openai").lower()

    if provider != "openai":
        raise NotImplementedError("Only WHISPER_PROVIDER='openai' is implemented right now.")

    model = getattr(settings, "WHISPER_MODEL", None) or "gpt-4o-transcribe"  # or 'whisper-1' if you prefer
    client = _build_openai_client()

    if not filepath or not Path(filepath).exists():
        raise RuntimeError(f"Audio file not found: {filepath}")

    try:
        with open(filepath, "rb") as f:
            # response_format="text" returns a raw string; some SDKs expose it at .text
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                response_format="text",
            )
        # New SDK: `resp` is already a plain string when response_format="text"
        return getattr(resp, "text", str(resp))
    except Exception as e:
        # Normalize into a friendly error without spilling stack traces
        raise RuntimeError(f"Transcription failed: {e}") from None
