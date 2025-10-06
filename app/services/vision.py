# app/services/vision.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import base64, io, os
import httpx

from app.config import settings


# ---------------- OpenAI client ----------------
def _build_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy, timeout=120) if proxy else None
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# ---------------- OCR helpers ----------------
def _preprocess_for_ocr(img):
    """
    PIL-only preproc that helps Tesseract on screenshots:
      - convert to L (grayscale)
      - optional autocontrast
      - upscale for small fonts
      - slight sharpen
      - binarize (threshold) if needed
    """
    from PIL import ImageOps, ImageFilter

    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=2)

    # Telegram compresses; upscale if small to help OCR
    min_side = min(g.size)
    if min_side < 900:
        scale = max(2, 900 // max(1, min_side))
        g = g.resize((g.width * scale, g.height * scale))

    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))

    # light binarization helps on gray UIs
    g = g.point(lambda p: 255 if p > 180 else 0)
    return g


def ocr_image(path: Path, lang: Optional[str] = None, psm: Optional[str] = None) -> str:
    """
    Stronger OCR with PIL pre-processing + Tesseract hints.
    Returns "" if pytesseract/Pillow not available or OCR fails.
    """
    if not path or not Path(path).exists():
        raise RuntimeError(f"Image not found: {path}")

    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""

    try:
        img = Image.open(path)
        img = _preprocess_for_ocr(img)

        lang = (lang or os.getenv("OCR_LANG") or "eng").strip()   # e.g. "spa+eng"
        psm = psm or os.getenv("OCR_PSM", "6")                    # 6 = uniform block of text
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        txt = pytesseract.image_to_string(img, lang=lang, config=cfg)
        return (txt or "").strip()
    except Exception:
        return ""


# ---------------- Vision helpers ----------------
def _img_to_data_url(path: Path) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def vision_extract_insights(path: Path, user_prompt: str = "", *, model: Optional[str] = None) -> str:
    """
    Generic vision extraction for tables/diagrams/signage.
    """
    if not path or not Path(path).exists():
        raise RuntimeError(f"Image not found: {path}")

    try:
        client = _build_openai_client()
    except Exception:
        return ""

    mdl = model or os.getenv("VISION_MODEL", "") or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    data_url = _img_to_data_url(path)
    if not data_url:
        return ""

    system = (
        "Eres un asistente técnico de RETIE. Extrae texto visible, títulos, listas "
        "y valores técnicos con fidelidad. Mantén términos tal como aparecen."
    )
    user_text = (user_prompt.strip() + "\n") if user_prompt else ""
    user_text += (
        "Lee el contenido de la imagen (texto/etiquetas/tabla/diagrama) y escribe un "
        "resumen textual limpio y estructurado con los puntos clave."
    )

    try:
        r = client.chat.completions.create(
            model=mdl,
            temperature=0,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def vision_extract_question(path: Path, *, model: Optional[str] = None) -> str:
    """
    Vision prompt specialized to return ONLY the question found in a screenshot.
    Useful when the image is a prompt/question screenshot.
    """
    try:
        client = _build_openai_client()
    except Exception:
        return ""

    mdl = model or os.getenv("VISION_MODEL", "") or "gpt-4o"
    data_url = _img_to_data_url(path)
    if not data_url:
        return ""

    system = (
        "Eres un extractor de preguntas. Devuelve únicamente la pregunta que aparece en la imagen, "
        "sin explicaciones, sin comillas y en una sola línea clara."
    )

    try:
        r = client.chat.completions.create(
            model=mdl,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Lee la imagen y escribe SOLO la pregunta capturada."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ---------------- Combo for question images ----------------
def extract_question_from_image(path: Path) -> str:
    """
    Strategy:
      1) OCR with spa+eng and PSM suited for single block
      2) Heuristic to pick the likely question line
      3) If unclear/empty → Vision with question-only prompt
    """
    # 1) OCR first (fast & cheap)
    txt = ocr_image(path, lang=os.getenv("OCR_LANG", "spa+eng"), psm=os.getenv("OCR_PSM", "6"))
    question = ""

    # 2) Simple heuristic: prefer the longest line that ends with '?'
    if txt:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        q_lines = [l for l in lines if l.endswith("?")]
        if q_lines:
            # pick the longest (often the full question)
            question = max(q_lines, key=len)
        else:
            # fall back to longest line if nothing ends with '?'
            question = max(lines, key=len) if lines else ""

    # 3) If still weak or very short → Vision fallback (more robust)
    if not question or len(question) < 15:
        question = vision_extract_question(path) or question

    return (question or "").strip()
