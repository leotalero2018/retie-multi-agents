"""E3 — Extracción de tablas con cascada de estrategias + registro de activos.

Cascada (de barata a cara) por cada tabla detectada:
  1. pdfplumber estrategia de LÍNEAS (rejilla vectorial)        → exacta, gratis
  2. pdfplumber estrategia de TEXTO sobre la región recortada    → gratis
  3. GPT-4o Vision sobre el render 300 dpi del recorte           → escaneos / tipográficas rotas
  4. SIEMPRE: PNG del recorte guardado como activo               → fallback de fidelidad perfecta

Cada tabla queda en el registro con ubicación exacta (doc, páginas, bbox),
PNG, JSON canónico (si se logró estructurar), markdown y estado.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pdfplumber

from .config import V2Config
from .registry import AssetRegistry, table_json_payload
from .vision import VisionBudget, extract_table_vision

log = logging.getLogger(__name__)

# Anclas de DEFINICIÓN de tabla: la línea EMPIEZA con "Tabla N" (las menciones
# inline "véase la Tabla N" no abren tabla).
_ANCHOR_RE = re.compile(r"^\s*Tabla\s+(\d+(?:[.\-]\d+)*(?:\([a-z]\))?)\b(.*)", re.IGNORECASE)
_CONT_RE = re.compile(r"continuaci[oó]n", re.IGNORECASE)

_MIN_ROWS = 2
_MIN_COLS = 2


# ──────────────────────────────────────────────────────────────────────────────
# Detección de anclas
# ──────────────────────────────────────────────────────────────────────────────
def find_anchors(pages_text: Dict[str, str]) -> List[Dict[str, Any]]:
    """[{tabla_id, page, titulo, is_continuation}] de líneas que abren tabla."""
    anchors: List[Dict[str, Any]] = []
    for pno_s, text in pages_text.items():
        for line in text.splitlines():
            m = _ANCHOR_RE.match(line)
            if not m:
                continue
            rest = (m.group(2) or "").strip(" .–-:")
            anchors.append({
                "tabla_id": m.group(1),
                "page": int(pno_s),
                "titulo": rest[:160],
                "is_continuation": bool(_CONT_RE.search(rest)),
            })
    anchors.sort(key=lambda a: (a["page"],))
    return anchors


def group_anchors(anchors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Agrupa por tabla_id: primera definición + páginas de continuación."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for a in anchors:
        tid = a["tabla_id"]
        g = grouped.get(tid)
        if g is None:
            grouped[tid] = {
                "tabla_id": tid,
                "titulo": a["titulo"] if not a["is_continuation"] else "",
                "pages": [a["page"]],
            }
        else:
            if a["page"] not in g["pages"]:
                g["pages"].append(a["page"])
            if not g["titulo"] and not a["is_continuation"]:
                g["titulo"] = a["titulo"]
    for g in grouped.values():
        g["pages"].sort()
        # Solo páginas contiguas a la definición cuentan como cuerpo de la tabla;
        # menciones lejanas en el documento son referencias, no la tabla.
        body = [g["pages"][0]]
        for p in g["pages"][1:]:
            if p <= body[-1] + 1:
                body.append(p)
        g["pages"] = body
    return grouped


# ──────────────────────────────────────────────────────────────────────────────
# Geometría: localizar la región de la tabla en la página
# ──────────────────────────────────────────────────────────────────────────────
def _anchor_rect(page: fitz.Page, tabla_id: str) -> Optional[fitz.Rect]:
    try:
        rects = page.search_for(f"Tabla {tabla_id}")
    except Exception:
        rects = []
    if not rects:
        return None
    # preferir la ocurrencia más a la izquierda (inicio de línea)
    return sorted(rects, key=lambda r: (r.x0, r.y0))[0]


def table_region(page: fitz.Page, tabla_id: str) -> fitz.Rect:
    """Región estimada de la tabla: desde el ancla hasta el final de página
    (recortada después si la geometría encuentra el bbox real)."""
    r = _anchor_rect(page, tabla_id)
    pr = page.rect
    if r is None:
        return fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y1)
    return fitz.Rect(pr.x0, max(pr.y0, r.y0 - 4), pr.x1, pr.y1)


def render_region_png(page: fitz.Page, rect: fitz.Rect, dpi: int) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    return pix.tobytes("png")


# ──────────────────────────────────────────────────────────────────────────────
# Extracción geométrica (pdfplumber)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_rows(data: List[List[Any]]) -> Tuple[List[str], List[List[str]]]:
    rows = [[("" if c is None else str(c).strip()) for c in row] for row in (data or [])]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return [], []
    headers = rows[0]
    body = rows[1:]
    ncol = len(headers)
    body = [(r + [""] * (ncol - len(r)))[:ncol] for r in body]
    return headers, body


def _consistent(headers: List[str], rows: List[List[str]]) -> bool:
    if len(headers) < _MIN_COLS or len(rows) < _MIN_ROWS:
        return False
    filled_headers = sum(1 for h in headers if h)
    if filled_headers < max(2, len(headers) // 2):
        return False
    # densidad de celdas con contenido
    cells = [c for r in rows for c in r]
    filled = sum(1 for c in cells if c)
    return filled / max(1, len(cells)) >= 0.45


def extract_geometric(pl_page, clip: Tuple[float, float, float, float]) -> Optional[Dict[str, Any]]:
    """Intenta lattice y luego text-strategy sobre la región recortada."""
    try:
        region = pl_page.crop(clip)
    except Exception:
        region = pl_page

    for metodo, settings in (
        ("lattice", {"vertical_strategy": "lines", "horizontal_strategy": "lines"}),
        ("text", {"vertical_strategy": "text", "horizontal_strategy": "text",
                  "text_x_tolerance": 2, "intersection_tolerance": 5}),
    ):
        try:
            found = region.find_tables(table_settings=settings)
        except Exception:
            continue
        if not found:
            continue
        best = max(found, key=lambda t: (t.bbox[3] - t.bbox[1]) * (t.bbox[2] - t.bbox[0]))
        try:
            headers, rows = _normalize_rows(best.extract())
        except Exception:
            continue
        if _consistent(headers, rows):
            return {"metodo": metodo, "headers": headers, "rows": rows,
                    "bbox": [round(v, 1) for v in best.bbox]}
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Markdown
# ──────────────────────────────────────────────────────────────────────────────
def to_markdown(titulo: str, headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    if titulo:
        lines.append(f"**{titulo}**")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        lines.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por documento
# ──────────────────────────────────────────────────────────────────────────────
def extract_tables_for_doc(cfg: V2Config, doc_meta: dict, pdf_path: Path,
                           inventory: dict, pages_text: Dict[str, str],
                           registry: AssetRegistry, budget: VisionBudget,
                           segments=None) -> Dict[str, int]:
    """Procesa todas las tablas del documento. Devuelve conteos por estado."""
    doc_id = doc_meta["doc_id"]
    page_profiles = {p["page"]: p for p in inventory["pages"]}
    anchors = find_anchors(pages_text)
    grouped = group_anchors(anchors)
    if not grouped:
        log.info("[E3] %s: sin anclas de tabla", doc_id)
        return {}

    # mapa página → artículo padre (si E2 ya corrió)
    page_to_articulo: Dict[int, str] = {}
    for seg in (segments or []):
        if seg.tipo in ("articulo", "numeral", "seccion"):
            for p in range(seg.page_start, seg.page_end + 1):
                page_to_articulo.setdefault(p, f"{seg.tipo} {seg.numero}")

    out_dir = cfg.assets_dir / "tables" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    fdoc = fitz.open(str(pdf_path))
    with pdfplumber.open(str(pdf_path)) as pdoc:
        for tid, info in grouped.items():
            try:
                estado = _process_table(cfg, doc_id, tid, info, fdoc, pdoc,
                                        page_profiles, out_dir, registry,
                                        budget, page_to_articulo)
            except Exception as exc:
                log.warning("[E3] %s tabla %s falló: %s", doc_id, tid, exc)
                estado = "revision"
            stats[estado] = stats.get(estado, 0) + 1
    fdoc.close()

    log.info("[E3] %s: %d tablas → %s", doc_id, sum(stats.values()),
             ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return stats


def _safe_id(tid: str) -> str:
    return re.sub(r"[^0-9A-Za-z.\-()]", "_", tid)


def _process_table(cfg: V2Config, doc_id: str, tid: str, info: dict,
                   fdoc: fitz.Document, pdoc, page_profiles: dict,
                   out_dir: Path, registry: AssetRegistry, budget: VisionBudget,
                   page_to_articulo: Dict[int, str]) -> str:
    pages: List[int] = info["pages"]
    first_page = pages[0]
    page = fdoc[first_page - 1]
    prof = page_profiles.get(first_page, {})
    scanned = prof.get("kind") == "scanned"

    region = table_region(page, tid)

    # PNG del recorte (siempre — es el fallback de fidelidad perfecta)
    png_bytes = render_region_png(page, region, cfg.table_dpi)
    png_parts = [png_bytes]
    for extra in pages[1:]:
        try:
            ep = fdoc[extra - 1]
            png_parts.append(render_region_png(ep, ep.rect, cfg.table_dpi))
        except Exception:
            pass

    hash_region = hashlib.sha256(b"".join(png_parts)).hexdigest()[:16]

    # Caché entre corridas: región idéntica + JSON previo → reutilizar.
    cached = registry.cached_extraction(doc_id, tid, hash_region)
    if cached:
        return cached["estado"]

    safe = _safe_id(tid)
    png_rel = f"tables/{doc_id}/{safe}.png"
    (out_dir / f"{safe}.png").write_bytes(png_bytes)
    for i, extra_png in enumerate(png_parts[1:], start=2):
        (out_dir / f"{safe}_p{i}.png").write_bytes(extra_png)

    # ── Cascada de extracción ────────────────────────────────────────────────
    geo = None
    if not scanned:
        clip = (region.x0, region.y0, region.x1, region.y1)
        geo = extract_geometric(pdoc.pages[first_page - 1], clip)

    vis = None
    need_vision = (geo is None) or scanned
    if need_vision and cfg.vision_enabled:
        vis = extract_table_vision(png_bytes, tid, cfg.vision_model, budget)

    headers: List[str] = []
    rows: List[List[str]] = []
    metodo = "none"
    estado = "solo_imagen"
    notes = None

    if geo and vis:
        gh, gr = geo["headers"], geo["rows"]
        vh, vr = [str(h) for h in vis["headers"]], [[str(c) for c in r] for r in vis["rows"]]
        if abs(len(gr) - len(vr)) <= max(1, int(0.1 * max(len(gr), len(vr)))):
            headers, rows, metodo, estado = vh, vr, "vision+geo", "verificada"
        else:
            headers, rows, metodo, estado = vh, vr, "vision", "revision"
            notes = f"geo={len(gr)} filas vs vision={len(vr)} filas"
    elif vis:
        headers = [str(h) for h in vis["headers"]]
        rows = [[str(c) for c in r] for r in vis["rows"]]
        metodo, estado = "vision", "extraida"
        notes = vis.get("notes")
        if vis.get("_truncated"):
            estado = "revision"
    elif geo:
        headers, rows, metodo, estado = geo["headers"], geo["rows"], geo["metodo"], "extraida"

    # Continuaciones multipágina con Vision (concatenar filas si headers compatibles)
    if estado in ("verificada", "extraida") and len(pages) > 1 and cfg.vision_enabled and metodo.startswith("vision"):
        for i, extra_png in enumerate(png_parts[1:], start=2):
            more = extract_table_vision(extra_png, tid, cfg.vision_model, budget)
            if more and len(more.get("headers", [])) == len(headers):
                rows.extend([[str(c) for c in r] for r in more["rows"]])

    json_rel = None
    markdown = None
    if headers and rows:
        json_rel = f"tables/{doc_id}/{safe}.json"
        (out_dir / f"{safe}.json").write_text(
            table_json_payload(info.get("titulo") or f"Tabla {tid}", headers, rows, notes),
            encoding="utf-8")
        markdown = to_markdown(info.get("titulo") or f"Tabla {tid}", headers, rows)

    registry.upsert_table({
        "doc_id": doc_id, "tabla_id": tid,
        "titulo": info.get("titulo") or f"Tabla {tid}",
        "pagina_inicio": first_page, "pagina_fin": pages[-1],
        "bbox": json.dumps([round(region.x0, 1), round(region.y0, 1),
                            round(region.x1, 1), round(region.y1, 1)]),
        "png_key": png_rel, "json_key": json_rel, "markdown": markdown,
        "estado": estado, "metodo": metodo,
        "articulo_padre": page_to_articulo.get(first_page),
        "hash_region": hash_region,
        "n_filas": len(rows), "n_cols": len(headers),
    })
    return estado
