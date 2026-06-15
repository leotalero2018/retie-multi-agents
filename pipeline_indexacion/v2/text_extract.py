"""E1 — Extracción de texto layout-aware.

Corrige los dos defectos medidos en el corpus real:
  1. NTC 2050 a dos columnas: el texto se extrae POR COLUMNA en orden de
     lectura (izquierda completa → derecha), con bloques full-width como
     separadores de banda. Antes (PyPDFLoader) las columnas se entrelazaban
     a mitad de oración.
  2. Headers/footers repetidos (p. ej. "RESOLUCIÓN NÚMERO 40117 ... Hoja N
     de 168"): se detectan automáticamente por frecuencia y se eliminan.

Salida: v2_work/text/{doc_id}.json  →  {"pages": {"1": "texto limpio", ...}}
"""
from __future__ import annotations

import bisect
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fitz

from .config import V2Config

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")
_PAGENUM_RE = re.compile(r"^\s*(?:p[áa]g(?:ina)?\.?\s*)?\d{1,4}\s*$", re.IGNORECASE)


def _norm_line(line: str) -> str:
    """Normaliza una línea para detección de boilerplate (dígitos → #)."""
    s = _WS_RE.sub(" ", line.strip().lower())
    return _DIGITS_RE.sub("#", s)


# ──────────────────────────────────────────────────────────────────────────────
# Orden de lectura por bandas y columnas
# ──────────────────────────────────────────────────────────────────────────────
def ordered_blocks(page: fitz.Page, n_columns: int, fullwidth_ratio: float) -> List[str]:
    """Bloques de texto en orden de lectura. Maneja layout a 2 columnas:
    los bloques full-width parten la página en bandas; dentro de cada banda
    se lee la columna izquierda completa y luego la derecha."""
    raw = [b for b in page.get_text("blocks") if b[6] == 0 and (b[4] or "").strip()]
    if not raw:
        return []
    if n_columns < 2:
        return [b[4] for b in sorted(raw, key=lambda b: (round(b[1], 1), b[0]))]

    width = page.rect.width
    fw = width * fullwidth_ratio
    mid = width / 2

    fulls = sorted([b for b in raw if (b[2] - b[0]) >= fw], key=lambda b: (b[1] + b[3]) / 2)
    fulls_y = [(b[1] + b[3]) / 2 for b in fulls]

    def sort_key(b) -> Tuple:
        yc = (b[1] + b[3]) / 2
        if (b[2] - b[0]) >= fw:
            # El bloque full-width ocupa su propia posición entre bandas.
            band = bisect.bisect_left(fulls_y, yc)
            return (band, 3, yc, b[0])
        band = bisect.bisect_right(fulls_y, yc)
        col = 0 if (b[0] + b[2]) / 2 < mid else 1
        return (band, col, yc, b[0])

    return [b[4] for b in sorted(raw, key=sort_key)]


# ──────────────────────────────────────────────────────────────────────────────
# Boilerplate (headers / footers repetidos)
# ──────────────────────────────────────────────────────────────────────────────
def detect_boilerplate(pages_lines: List[List[str]], freq_threshold: float) -> set[str]:
    """Líneas (normalizadas) que se repiten al inicio/fin de muchas páginas."""
    n_pages = max(1, len(pages_lines))
    counter: Counter[str] = Counter()
    for lines in pages_lines:
        edge = lines[:4] + lines[-3:]
        seen_in_page = set()
        for ln in edge:
            norm = _norm_line(ln)
            if len(norm) >= 6 and norm not in seen_in_page:
                seen_in_page.add(norm)
                counter[norm] += 1
    return {norm for norm, c in counter.items() if c / n_pages >= freq_threshold}


def _clean_page(lines: List[str], boilerplate: set[str]) -> str:
    out: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            out.append("")
            continue
        if _norm_line(stripped) in boilerplate:
            continue
        if _PAGENUM_RE.match(stripped):
            continue
        out.append(stripped)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# API por documento
# ──────────────────────────────────────────────────────────────────────────────
def extract_doc_text(cfg: V2Config, doc_meta: dict, pdf_path: Path,
                     inventory: dict, *, force: bool = False) -> Dict[str, str]:
    """Extrae y limpia el texto de todas las páginas. Persiste y devuelve
    {pagina(str): texto}."""
    doc_id = doc_meta["doc_id"]
    out_path = cfg.text_dir / f"{doc_id}.json"

    if out_path.exists() and not force:
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("sha256") == inventory.get("sha256"):
                log.info("[E1] %s sin cambios — texto reutilizado", doc_id)
                return prev["pages"]
        except Exception:
            pass

    page_profiles = {p["page"]: p for p in inventory["pages"]}
    doc = fitz.open(str(pdf_path))

    # Pasada 1: bloques ordenados por página → líneas
    pages_lines: List[List[str]] = []
    for i in range(len(doc)):
        prof = page_profiles.get(i + 1, {})
        blocks = ordered_blocks(doc[i], prof.get("n_columns", 1), cfg.fullwidth_ratio)
        lines: List[str] = []
        for blk in blocks:
            lines.extend(blk.splitlines())
        pages_lines.append(lines)
    doc.close()

    # Pasada 2: boilerplate global del documento
    boilerplate = detect_boilerplate(pages_lines, cfg.boilerplate_freq)
    if boilerplate:
        log.info("[E1] %s: %d líneas de header/footer detectadas y eliminadas",
                 doc_id, len(boilerplate))

    pages: Dict[str, str] = {}
    for i, lines in enumerate(pages_lines):
        pages[str(i + 1)] = _clean_page(lines, boilerplate)

    out_path.write_text(
        json.dumps({"doc_id": doc_id, "sha256": inventory.get("sha256"),
                    "boilerplate": sorted(boilerplate), "pages": pages},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    total_chars = sum(len(t) for t in pages.values())
    log.info("[E1] %s: %d páginas extraídas (%d chars)", doc_id, len(pages), total_chars)
    return pages


def load_doc_text(cfg: V2Config, doc_id: str) -> Dict[str, str] | None:
    p = cfg.text_dir / f"{doc_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["pages"]
