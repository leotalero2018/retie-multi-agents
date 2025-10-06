# app/services/vision.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import base64, io, os
import httpx

from app.config import settings


# ---------- OpenAI client (shared) ----------
def _build_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy, timeout=120) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# ---------- Local OCR (fast & cheap) ----------
def ocr_image(path: Path, lang: Optional[str] = None) -> str:
    """
    Extract text from an image using Tesseract.
    - lang: e.g., "eng" or "spa". If None, tries env OCR_LANG (default: "eng").
    - Returns "" on OCR failure (so caller can choose a fallback).
    """
    if not path or not Path(path).exists():
        raise RuntimeError(f"Image not found: {path}")

    # Lazy imports so module import never crashes if OCR libs are missing
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""  # OCR optional; let caller decide to fallback to Vision

    try:
        lang = (lang or os.getenv("OCR_LANG") or "eng").strip()
        img = Image.open(path)

        # Gentle pre-processing: grayscale tends to help
        img = img.convert("L")

        # Page segmentation mode 6: Assume a uniform block of text
        txt = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
        return (txt or "").strip()
    except Exception:
        return ""


# ---------- Optional OpenAI Vision (for diagrams/tables/non-text) ----------
def _img_to_data_url(path: Path) -> str:
    """
    Convert image to PNG data URL. Returns empty string if Pillow not present.
    """
    try:
        from PIL import Image
    except Exception:
        return ""

    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def vision_extract_insights(path: Path, user_prompt: str = "", *, model: Optional[str] = None) -> str:
    """
    Ask a vision model to read the image (tables, diagrams/signage) and return useful text.
    Cheap default: gpt-4o-mini.
    Returns "" if Vision cannot be called (e.g., no key) or if request fails.
    """
    if not path or not Path(path).exists():
        raise RuntimeError(f"Image not found: {path}")

    try:
        client = _build_openai_client()
    except Exception:
        return ""

    mdl = model or os.getenv("VISION_MODEL", "") or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    img_data_url = _img_to_data_url(path)
    if not img_data_url:
        return ""

    # Compose precise, RAG-friendly instruction
    system = (
        "Eres un asistente técnico de RETIE. Extrae y normaliza texto visible, "
        "tablas y señales relevantes. Mantén términos técnicos tal como aparecen."
    )
    prompt_parts = []
    if user_prompt.strip():
        prompt_parts.append(user_prompt.strip())
    prompt_parts.append(
        "Lee el contenido de la imagen (texto/etiquetas/tabla/diagrama) y devuélveme "
        "un resumen textual estructurado y útil para RAG. Si hay valores numéricos o límites, "
        "inclúyelos tal cual."
    )
    user = "\n".join(prompt_parts)

    try:
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
            max_tokens=500,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ---------- Optional combo helper (OCR first, Vision fallback) ----------
def extract_text_from_image(path: Path, user_prompt: str = "", *, ocr_lang: Optional[str] = None) -> str:
    """
    Try OCR first; if empty or short, fallback to Vision model.
    """
    text = ocr_image(path, lang=ocr_lang)
    if text and len(text) >= 20:  # small heuristic
        return text
    # Fallback to vision for diagrams/low-OCR images
    vision = vision_extract_insights(path, user_prompt=user_prompt)
    # Prefer vision if OCR was empty/short
    return vision or text
