# app/services/vision.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import base64, io, os
import httpx
from PIL import Image
import pytesseract

from app.config import settings

# ---------- Local OCR (fast & cheap) ----------
def ocr_image(path: Path, lang: Optional[str] = None) -> str:
    """
    Extract text from an image using Tesseract.
    - lang: e.g., "eng" or "spa". If None, tries env OCR_LANG (default: "eng").
    """
    lang = lang or os.getenv("OCR_LANG", "eng")
    img = Image.open(path)
    # A gentle pre-process tends to help: convert to grayscale
    img = img.convert("L")
    txt = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
    return (txt or "").strip()

# ---------- Optional OpenAI Vision (for non-text content / diagrams) ----------
def _build_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy, timeout=60) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

def _img_to_data_url(path: Path) -> str:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def vision_extract_insights(path: Path, user_prompt: str = "", *, model: Optional[str] = None) -> str:
    """
    Ask a vision model to read the image (tables, diagrams/signage) and return useful text.
    Cheap default: gpt-4o-mini.
    """
    client = _build_openai_client()
    mdl = model or os.getenv("VISION_MODEL", "gpt-4o-mini")
    img_data_url = _img_to_data_url(path)

    # Compose a precise instruction that helps RAG
    system = "Extrae y normaliza texto visible y notas técnicas útiles para un asistente RETIE."
    user = (
        (user_prompt.strip() + "\n") if user_prompt else "" +
        "Lee el contenido de la imagen (texto/etiquetas/tabla). "
        "Devuélveme un resumen textual limpio, con términos técnicos tal como aparecen."
    )

    resp = client.chat.completions.create(
        model=mdl,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": img_data_url}},
            ]},
        ],
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()
