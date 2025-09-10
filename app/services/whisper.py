# app/services/whisper.py
from pathlib import Path
from typing import Optional
from app.config import settings

def transcribe_audio(filepath: Path, language: Optional[str] = "es") -> str:
    """
    Transcribe un archivo de audio a texto. Soporta provider 'openai' (default).
    filepath: ruta local (wav/mp3/ogg/opus/m4a...)
    """
    provider = (settings.WHISPER_PROVIDER or "openai").lower()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        with open(filepath, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=settings.WHISPER_MODEL,
                file=f,                # auto-detecta formato
                language=language,     # sugiere idioma
                response_format="text" # devuelve string plano
            )
        # SDK nuevo devuelve .text si response_format="text"
        return getattr(resp, "text", str(resp))

    # (Opcional) Proveedor local: faster-whisper, etc. -> a futuro
    raise NotImplementedError("WHISPER_PROVIDER local no implementado aún.")
