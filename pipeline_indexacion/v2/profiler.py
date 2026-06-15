"""E0 — Perfilado por página, inventario e incrementalidad.

Para cada PDF produce un inventario JSON con:
  - sha256 del archivo (skip de documentos sin cambios)
  - por página: tipo (native|scanned|mixed), nº de columnas, hint de tabla,
    bboxes de imágenes grandes y hash del texto (incrementalidad fina).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

from .config import V2Config

log = logging.getLogger(__name__)

_TABLE_HINT_RE = re.compile(r"\bTabla\s+\d", re.IGNORECASE)
_FIGURE_HINT_RE = re.compile(r"\bFigura\s+[\dA-Z]", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def detect_columns(page: fitz.Page, fullwidth_ratio: float) -> int:
    """Detecta 1 o 2 columnas por la distribución X de los bloques de texto."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and (b[4] or "").strip()]
    if len(blocks) < 6:
        return 1
    width = page.rect.width
    fw = width * fullwidth_ratio
    mid = width / 2
    nonfull = [b for b in blocks if (b[2] - b[0]) < fw]
    if len(nonfull) < 6:
        return 1
    left = [b for b in nonfull if (b[0] + b[2]) / 2 < mid]
    right = [b for b in nonfull if (b[0] + b[2]) / 2 >= mid]
    if min(len(left), len(right)) >= max(2, int(0.2 * len(nonfull))):
        return 2
    return 1


def _image_rects(page: fitz.Page) -> List[Dict[str, Any]]:
    """Rects de imágenes raster en la página, con su xref (para dedup)."""
    out: List[Dict[str, Any]] = []
    for im in page.get_images(full=True):
        xref = im[0]
        try:
            for r in page.get_image_rects(xref):
                out.append({"xref": xref, "bbox": [round(r.x0, 1), round(r.y0, 1),
                                                   round(r.x1, 1), round(r.y1, 1)]})
        except Exception:
            continue
    return out


def profile_doc(cfg: V2Config, doc_meta: dict, pdf_path: Path) -> Dict[str, Any]:
    """Perfila un documento completo. Devuelve el inventario (y lo persiste)."""
    doc_id = doc_meta["doc_id"]
    file_hash = sha256_file(pdf_path)

    inv_path = cfg.inventory_dir / f"{doc_id}.json"
    if inv_path.exists():
        try:
            prev = json.loads(inv_path.read_text(encoding="utf-8"))
            if prev.get("sha256") == file_hash:
                log.info("[E0] %s sin cambios (sha256 igual) — inventario reutilizado", doc_id)
                return prev
        except Exception:
            pass

    doc = fitz.open(str(pdf_path))
    pages: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc[i]
        parea = page.rect.width * page.rect.height or 1.0
        text = page.get_text("text") or ""
        imgs = _image_rects(page)

        cover_max = 0.0
        big_imgs = []
        for im in imgs:
            x0, y0, x1, y1 = im["bbox"]
            cover = ((x1 - x0) * (y1 - y0)) / parea
            cover_max = max(cover_max, cover)
            if cover >= cfg.bigimg_cover:
                big_imgs.append({**im, "cover": round(cover, 3)})

        if cover_max >= cfg.scanned_cover:
            kind = "scanned"
        elif cover_max >= 0.30:
            kind = "mixed"
        else:
            kind = "native"

        pages.append({
            "page": i + 1,
            "kind": kind,
            "n_columns": detect_columns(page, cfg.fullwidth_ratio) if kind != "scanned" else 1,
            "has_table_hint": bool(_TABLE_HINT_RE.search(text)),
            "has_figure_hint": bool(_FIGURE_HINT_RE.search(text)),
            "big_images": big_imgs,
            "text_chars": len(text.strip()),
            "text_hash": _text_hash(text),
            "width": round(page.rect.width, 1),
            "height": round(page.rect.height, 1),
        })
    doc.close()

    inventory = {
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "sha256": file_hash,
        "n_pages": len(pages),
        "pages": pages,
        "summary": {
            "native": sum(1 for p in pages if p["kind"] == "native"),
            "scanned": sum(1 for p in pages if p["kind"] == "scanned"),
            "mixed": sum(1 for p in pages if p["kind"] == "mixed"),
            "two_columns": sum(1 for p in pages if p["n_columns"] == 2),
            "table_hint_pages": sum(1 for p in pages if p["has_table_hint"]),
        },
    }
    inv_path.write_text(json.dumps(inventory, ensure_ascii=False), encoding="utf-8")
    s = inventory["summary"]
    log.info("[E0] %s: %d págs (native=%d scanned=%d mixed=%d, 2col=%d, tabla-hint=%d)",
             doc_id, len(pages), s["native"], s["scanned"], s["mixed"],
             s["two_columns"], s["table_hint_pages"])
    return inventory


def load_inventory(cfg: V2Config, doc_id: str) -> Dict[str, Any] | None:
    p = cfg.inventory_dir / f"{doc_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
