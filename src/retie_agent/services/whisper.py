# app/services/whisper.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import os, httpx, re

from retie_agent.config import settings

_DOMAIN_PROMPT_ES = (
    # Keep this concise (200–300 chars). Add key acronyms & proper nouns that matter.
    "Dominio: normativa eléctrica colombiana. Palabras frecuentes: RETIE, "
    "Resolución 40117, NTC 2050, puesta a tierra, cortocircuito, sobretensión, "
    "protección diferencial, subestación, transformador, conductor, neutro, fase."
)

def _build_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or settings.")
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
    http_client = httpx.Client(proxies=proxy, timeout=120) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

def _looks_bad(text: str) -> bool:
    """Heuristic to detect suspicious transcripts (very short or mostly non-letters)."""
    t = (text or "").strip()
    if len(t) < 8:
        return True
    letters = len(re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", t))
    return letters < max(8, len(t) * 0.3)

def transcribe_audio(filepath: Path, language: Optional[str] = "es") -> str:
    """
    Transcribe with whisper-1; if the result looks poor, retry with gpt-4o-transcribe.
    Uses a domain prompt to bias recognition toward RETIE vocabulary.
    Returns plain text, raises RuntimeError on failure.
    """
    provider = (getattr(settings, "WHISPER_PROVIDER", "openai") or "openai").lower()
    if provider != "openai":
        raise NotImplementedError("Only WHISPER_PROVIDER='openai' is implemented.")

    primary_model = getattr(settings, "WHISPER_MODEL", None) or "whisper-1"
    fallback_model = os.getenv("WHISPER_FALLBACK_MODEL", "gpt-4o-transcribe")

    client = _build_openai_client()
    p = Path(filepath)
    if not p.exists():
        raise RuntimeError(f"Audio file not found: {filepath}")

    def _call(model: str) -> str:
        with open(p, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                # KEY TUNING:
                temperature=0,
                prompt=_DOMAIN_PROMPT_ES,
                response_format="text",
            )
        return getattr(resp, "text", str(resp))

    try:
        text = _call(primary_model).strip()
        if _looks_bad(text) and fallback_model and fallback_model != primary_model:
            # Retry once with a stronger model
            text2 = _call(fallback_model).strip()
            if not _looks_bad(text2):
                return text2
        return text
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from None
