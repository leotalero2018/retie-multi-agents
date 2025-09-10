from pathlib import Path
from typing import Optional, List, Tuple

import pdfplumber
from pptx import Presentation
from PIL import Image
import pytesseract

from app.utils.text import clean_text


# === Extracción "completa" (compat) ===

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            text_parts.append(txt)
    return clean_text("\n".join(text_parts))


def extract_text_from_pptx(path: Path) -> str:
    prs = Presentation(str(path))
    parts = []
    for slide in prs.slides:
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text.append(shape.text)
        parts.append("\n".join(slide_text))
    return clean_text("\n".join(parts))


def extract_text_from_image(path: Path, lang: Optional[str] = "spa+eng") -> str:
    img = Image.open(str(path))
    raw = pytesseract.image_to_string(img, lang=lang)
    return clean_text(raw)


# === NUEVO: extracción por "páginas" (PDF=page, PPTX=slide, IMG/TXT=1) ===
def extract_pages(path: Path, ocr_lang: str = "spa+eng") -> List[Tuple[int, str]]:
    """
    Devuelve lista de (page_num, text) ya limpiado:
      - PDF: una tupla por página
      - PPT/PPTX: una tupla por slide
      - IMG/TXT: una sola tupla (1, texto)
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        out: List[Tuple[int, str]] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                txt = clean_text(txt)
                if txt:
                    out.append((i, txt))
        return out

    if ext in {".ppt", ".pptx"}:
        prs = Presentation(str(path))
        out: List[Tuple[int, str]] = []
        for i, slide in enumerate(prs.slides, start=1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)
            txt = clean_text("\n".join(slide_text))
            if txt:
                out.append((i, txt))
        return out

    if ext in {".png", ".jpg", ".jpeg"}:
        txt = extract_text_from_image(path, lang=ocr_lang)
        return [(1, txt)] if txt else []

    if ext == ".txt":
        txt = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        return [(1, txt)] if txt else []

    raise ValueError(f"Extensión no soportada: {ext}")
