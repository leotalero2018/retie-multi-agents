# app/services/vision.py
from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
from retie_agent.config import settings
from retie_agent.llm.provider import create_chat_completion


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
        r = create_chat_completion(
            client,
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
        r = create_chat_completion(
            client,
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
        re.search(
            r"(pregunta|verdadero|falso|marque|seleccione|opción\s+[A-D]|[a-d]\.)",
            txt,
            re.IGNORECASE,
        )
    )


def analyze_question_image(path: Path, agent_key: str) -> Dict[str, Any]:
    """
    Improved RETIE evaluator:
    Detects user's selected answer (A/B/C/D or Verdadero/Falso),
    queries RETIE context, and outputs:
        'La respuesta correcta es X. Según el RETIE, ...'
    """

    from retie_agent.retriever.retrieve import search
    from retie_agent.agent.retie_agent import _resolve_collection

    # --- 1️⃣ Extract question text ---
    question_text = extract_question_from_image(path)
    if not question_text:
        return {"error": "No se encontró ninguna pregunta legible."}

    # --- 2️⃣ OCR full text for options ---
    txt_raw = ocr_image(path, lang="spa+eng")

    # Normalize characters
    txt_raw = re.sub(r"[·•▪]", ".", txt_raw)
    txt_raw = re.sub(r"\s+", " ", txt_raw)

    # --- 3️⃣ Detect user's answer ---
    user_answer = ""
    for opt in ["A", "B", "C", "D", "TRUE", "FALSE", "VERDADERO", "FALSO"]:
        if re.search(rf"\b{opt}\b", txt_raw, re.IGNORECASE):
            user_answer = opt.upper()
            break

    # --- 4️⃣ Detect available options in text (A., B., C., D.) ---
    option_lines = re.findall(r"([A-Da-d]\s*[\).\-:]\s*[^A-Da-d]+)", txt_raw)
    options_formatted = "\n".join(option_lines) if option_lines else ""

    # --- 5️⃣ Retrieve supporting RETIE context ---
    coll = _resolve_collection(agent_key, explicit=None)
    hits = search(question_text, top_k=3, collection_name=coll)
    if not hits:
        return {
            "question": question_text,
            "user_answer": user_answer or "?",
            "expected_answer": "?",
            "is_correct": False,
            "explanation": "No se encontró referencia en la base RETIE.",
        }

    refs = []
    for h in hits:
        ctx = h.get("page_content", "")
        meta = h.get("meta", {})
        src = meta.get("source", "")
        page = meta.get("page_number", "")
        refs.append(f"{ctx}\n(Fuente: {os.path.basename(src)} pág. {page})")
    context_text = "\n\n".join(refs[:3])

    # --- 6️⃣ Ask model for concise correct answer + justification ---
    try:
        client = _build_openai_client()
        mdl = os.getenv("QA_REASONING_MODEL") or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")

        system = (
            "Eres un examinador experto en el reglamento RETIE. "
            "Analiza la pregunta y el contexto técnico. "
            "Si hay opciones múltiples, elige UNA sola (A, B, C, D) o Verdadero/Falso. "
            "Tu salida DEBE seguir exactamente este formato:\n\n"
            "La respuesta correcta es <A/B/C/D o Verdadero/Falso>.\n"
            "Según el RETIE, <explicación breve y técnica basada en el texto>.\n\n"
            "No incluyas títulos, saludos, numeraciones ni texto adicional."
        )

        user_prompt = (
            f"Pregunta detectada:\n{question_text}\n\n"
            f"Opciones detectadas:\n{options_formatted or 'No claras'}\n\n"
            f"Respuesta marcada por el usuario: {user_answer or 'Desconocida'}\n\n"
            f"Fragmentos del RETIE:\n{context_text}"
        )

        r = create_chat_completion(
            client,
            model=mdl,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )

        reasoning = (r.choices[0].message.content or "").strip()

        # --- 7️⃣ Clean format ---
        clean_text = re.sub(
            r"^(✅|❌|\*\*|#|\s*Pregunta:.*|Tu respuesta:.*|Respuesta esperada:.*)",
            "",
            reasoning,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()

        # If model returns just “B” or “Verdadero”, normalize it
        if not clean_text.lower().startswith("la respuesta correcta"):
            clean_text = f"La respuesta correcta es {clean_text.strip('.')}."

        return {
            "question": question_text,
            "user_answer": user_answer or "?",
            "explanation": clean_text,
        }

    except Exception as e:
        return {
            "question": question_text,
            "user_answer": user_answer or "?",
            "explanation": f"No se pudo generar explicación ({e}).",
        }
