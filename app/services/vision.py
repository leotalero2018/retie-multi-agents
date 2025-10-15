# app/services/vision.py
from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any

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
    """PIL pre-processing for OCR robustness."""
    from PIL import ImageOps, ImageFilter

    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=2)

    # Upscale if small (Telegram often compresses)
    min_side = min(g.size)
    if min_side < 900:
        scale = max(2, 900 // max(1, min_side))
        g = g.resize((g.width * scale, g.height * scale))

    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    g = g.point(lambda p: 255 if p > 180 else 0)
    return g


def ocr_image(path: Path, lang: Optional[str] = None, psm: Optional[str] = None) -> str:
    """Strong OCR using PIL + pytesseract."""
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
        lang = (lang or os.getenv("OCR_LANG") or "eng+spa").strip()
        psm = psm or os.getenv("OCR_PSM", "6")
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
    """Generic Vision OCR/summary fallback."""
    if not path or not Path(path).exists():
        raise RuntimeError(f"Image not found: {path}")
    try:
        client = _build_openai_client()
    except Exception:
        return ""

    mdl = model or os.getenv("VISION_MODEL") or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    data_url = _img_to_data_url(path)
    if not data_url:
        return ""

    system = (
        "Eres un asistente técnico de RETIE. Extrae texto visible, títulos, listas "
        "y valores técnicos con fidelidad. Mantén los términos tal como aparecen."
    )
    user_text = (user_prompt.strip() + "\n") if user_prompt else ""
    user_text += (
        "Lee el contenido de la imagen y escribe un resumen textual limpio y estructurado."
    )

    try:
        r = client.chat.completions.create(
            model=mdl,
            temperature=0,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def vision_extract_question(path: Path, *, model: Optional[str] = None) -> str:
    """Extracts only the question from an image."""
    try:
        client = _build_openai_client()
    except Exception:
        return ""

    mdl = model or os.getenv("VISION_MODEL") or "gpt-4o"
    data_url = _img_to_data_url(path)
    if not data_url:
        return ""

    system = (
        "Eres un extractor de preguntas. Devuelve únicamente la pregunta de la imagen, "
        "sin explicaciones ni comillas, en una sola línea clara."
    )

    try:
        r = client.chat.completions.create(
            model=mdl,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Lee la imagen y escribe SOLO la pregunta capturada."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def extract_question_from_image(path: Path) -> str:
    """Combined OCR + vision approach to extract a clean question."""
    txt = ocr_image(path, lang="spa+eng", psm="6")
    question = ""
    if txt:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        q_lines = [l for l in lines if l.endswith("?")]
        if q_lines:
            question = max(q_lines, key=len)
        else:
            question = max(lines, key=len) if lines else ""
    if not question or len(question) < 15:
        question = vision_extract_question(path) or question
    return (question or "").strip()


# ---------------- Smart QA detection ----------------
def is_question_image(txt: str) -> bool:
    """Detect if OCR text likely belongs to a question image."""
    return bool(
        re.search(r"(pregunta|verdadero|falso|marque|seleccione|opción\s+[A-D])", txt, re.IGNORECASE)
    )


def analyze_question_image(path: Path, agent_key: str) -> Dict[str, Any]:
    """
    Enhanced QA evaluator for RETIE questions.
    Produces structured, contextual feedback (A/B/C/D or True/False)
    with justification citing the RETIE reference text.
    """
    from app.retriever.retrieve import search
    from app.agent.retie_agent import _resolve_collection

    question_text = extract_question_from_image(path)
    if not question_text:
        return {"error": "No se encontró ninguna pregunta legible."}

    # --- Detect user's marked answer ---
    txt_raw = ocr_image(path, lang="spa+eng")
    user_answer = ""
    for opt in ["A", "B", "C", "D", "TRUE", "FALSE", "VERDADERO", "FALSO"]:
        if re.search(rf"\b{opt}\b", txt_raw, re.IGNORECASE):
            user_answer = opt.upper()
            break

    # --- Retrieve supporting context from Chroma ---
    coll = _resolve_collection(agent_key, explicit=None)
    hits = search(question_text, top_k=3, collection_name=coll)
    if not hits:
        return {
            "question": question_text,
            "user_answer": user_answer or "?",
            "expected_answer": "?",
            "is_correct": False,
            "explanation": "No se encontró referencia en la base de conocimiento RETIE.",
            "verdict": "❌ Sin referencia",
        }

    refs = []
    for h in hits:
        ctx = h.get("page_content", "")
        meta = h.get("meta", {})
        src = meta.get("source", "")
        page = meta.get("page_number", "")
        refs.append(f"Documento: {os.path.basename(src)} (pág. {page})\n{ctx}")
    context_text = "\n\n".join(refs[:3])

    # --- Query OpenAI for reasoning ---
    try:
        client = _build_openai_client()
        mdl = os.getenv("QA_REASONING_MODEL") or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")

        system = (
            "Eres un evaluador experto en el reglamento RETIE. "
            "Analiza la pregunta y la respuesta marcada, usando los fragmentos del RETIE proporcionados. "
            "Responde SIEMPRE en el formato:\n\n"
            "1. Respuesta correcta o incorrecta: <Correcta/Incorrecta/Parcialmente correcta>\n"
            "2. Respuesta correcta: <A/B/C/D/Verdadero/Falso>\n"
            "3. Artículo o referencia: <si existe>\n"
            "4. Justificación: <breve explicación basada en el texto>\n\n"
            "Sé preciso, cita artículos o secciones si aparecen, y usa tono institucional."
        )

        user = (
            f"Pregunta: {question_text}\n"
            f"Respuesta del usuario: {user_answer or 'Desconocida'}\n\n"
            f"Fragmentos del RETIE y contexto:\n{context_text}"
        )

        r = client.chat.completions.create(
            model=mdl,
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        reasoning = (r.choices[0].message.content or "").strip()

        # --- Parse model output ---
        correct_re = re.search(r"respuesta\s+correcta\s*:\s*(\w+)", reasoning, re.IGNORECASE)
        expected = correct_re.group(1).upper() if correct_re else "?"
        status_re = re.search(r"(correcta|incorrecta|parcial)", reasoning, re.IGNORECASE)
        status = status_re.group(1).lower() if status_re else "incorrecta"
        is_correct = "correcta" in status and "parcial" not in status
        verdict = (
            "✅ Correcta" if is_correct else
            ("🟡 Parcialmente correcta" if "parcial" in status else "❌ Incorrecta")
        )

        # Extract article if present
        art_match = re.search(r"(art[ií]culo\s+\d+[.\d]*)", reasoning, re.IGNORECASE)
        article = art_match.group(1) if art_match else ""

        # Extract justification text
        justif_match = re.search(r"(?:(?:justificación|explicación)[:\-]\s*)(.*)", reasoning, re.IGNORECASE | re.DOTALL)
        justification = justif_match.group(1).strip() if justif_match else reasoning

        # Build explanation paragraph
        explanation = (
            f"La respuesta correcta es {expected}. "
            f"{('Según ' + article + ', ' if article else 'Según el RETIE, ')}"
            f"{justification}"
        )

        return {
            "question": question_text,
            "user_answer": user_answer or "?",
            "expected_answer": expected,
            "is_correct": is_correct or "parcial" in status,
            "explanation": explanation,
            "verdict": verdict,
        }

    except Exception as e:
        return {
            "question": question_text,
            "user_answer": user_answer or "?",
            "expected_answer": "?",
            "is_correct": False,
            "explanation": f"No se pudo generar explicación ({e}).",
            "verdict": "❌ Error",
        }
