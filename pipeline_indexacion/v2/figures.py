"""E4 — Extracción de figuras y diagramas.

- Candidatas: imágenes grandes (>8% del área) en páginas nativas/mixtas.
- El escaneo página-completa de los libros RETIE NO es una figura (se filtra
  por cobertura >70% en página 'scanned').
- Ruido (logos/marcas de agua repetidos en cientos de páginas) se descarta
  deduplicando por hash del XObject.
- Cada figura: PNG en assets/, caption por GPT-4o Vision (opcional, cacheada
  por hash_region) y registro con ubicación exacta.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import fitz

from .config import V2Config
from .registry import AssetRegistry
from .vision import VisionBudget, caption_figure_vision

log = logging.getLogger(__name__)

_FIG_ID_RE = re.compile(r"\bFigura\s+([0-9]+(?:[.\-][0-9A-Za-z()]+)*)", re.IGNORECASE)


def _xref_hash(doc: fitz.Document, xref: int) -> Optional[str]:
    try:
        img = doc.extract_image(xref)
        return hashlib.sha256(img["image"]).hexdigest()[:16]
    except Exception:
        return None


def extract_figures_for_doc(cfg: V2Config, doc_meta: dict, pdf_path: Path,
                            inventory: dict, pages_text: Dict[str, str],
                            registry: AssetRegistry, budget: VisionBudget) -> int:
    doc_id = doc_meta["doc_id"]
    n_pages = inventory["n_pages"]
    doc = fitz.open(str(pdf_path))

    # 1) Censo de hashes para filtrar imágenes repetidas (logos, marcas de agua)
    hash_count: Counter[str] = Counter()
    page_candidates: List[tuple[int, dict, str]] = []  # (page, big_image, hash)
    for prof in inventory["pages"]:
        if prof["kind"] == "scanned":
            continue  # el escaneo de página completa no es figura
        for im in prof.get("big_images", []):
            if im.get("cover", 0) >= cfg.scanned_cover:
                continue
            h = _xref_hash(doc, im["xref"])
            if h is None:
                continue
            hash_count[h] += 1
            page_candidates.append((prof["page"], im, h))

    boiler_threshold = max(3, int(0.05 * n_pages))
    boiler = {h for h, c in hash_count.items() if c > boiler_threshold}

    out_dir = cfg.assets_dir / "figures" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    seen_ids: set[str] = set()
    for pno, im, h in page_candidates:
        if h in boiler:
            continue
        x0, y0, x1, y1 = im["bbox"]
        rect = fitz.Rect(x0, y0, x1, y1)
        if rect.width < 60 or rect.height < 60:
            continue

        page = doc[pno - 1]
        # figura_id desde el texto cercano ("Figura 250-2") o sintético
        page_txt = pages_text.get(str(pno), "")
        m = _FIG_ID_RE.search(page_txt)
        figura_id = m.group(1) if m else f"p{pno}_{im['xref']}"
        if figura_id in seen_ids:
            figura_id = f"{figura_id}_x{im['xref']}"
        seen_ids.add(figura_id)

        try:
            mat = fitz.Matrix(cfg.figure_dpi / 72, cfg.figure_dpi / 72)
            png_bytes = page.get_pixmap(matrix=mat, clip=rect).tobytes("png")
        except Exception as exc:
            log.debug("[E4] %s figura p%d render falló: %s", doc_id, pno, exc)
            continue

        hash_region = hashlib.sha256(png_bytes).hexdigest()[:16]
        prev = registry.get_figure(doc_id, figura_id)
        caption = prev.get("caption") if (prev and prev.get("hash_region") == hash_region) else None

        safe = re.sub(r"[^0-9A-Za-z.\-()_]", "_", figura_id)
        png_rel = f"figures/{doc_id}/{safe}.png"
        (out_dir / f"{safe}.png").write_bytes(png_bytes)

        if caption is None and cfg.vision_enabled and cfg.vision_caption_figures:
            caption = caption_figure_vision(png_bytes, cfg.vision_model, budget)
        if not caption:
            caption = f"Figura {figura_id} ({doc_meta.get('doc_name', doc_id)}, pág. {pno})"

        registry.upsert_figure({
            "doc_id": doc_id, "figura_id": figura_id, "caption": caption,
            "pagina": pno, "bbox": json.dumps([x0, y0, x1, y1]),
            "png_key": png_rel, "hash_region": hash_region,
        })
        count += 1

    doc.close()
    log.info("[E4] %s: %d figuras registradas (%d imágenes descartadas como ruido)",
             doc_id, count, sum(c for h, c in hash_count.items() if h in boiler))
    return count
