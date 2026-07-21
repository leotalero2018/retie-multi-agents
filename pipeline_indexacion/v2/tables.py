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
# El ID debe capturar el SUFIJO DE LETRA: RETIE usa "Tabla 2.3.26.2.2.1.a" y
# NTC "Tabla 392.10 (A)". Truncarlo fusionaba las tablas .a/.b bajo un mismo
# id (solo sobrevivía la primera región → se extraía la tabla equivocada) y
# rompía el lookup canónico por identificador en runtime.
# El sufijo ".a" se exige en minúscula ((?-i:...)) para no capturar ".A" de un
# título que empiece con mayúscula pegada al punto.
_ANCHOR_RE = re.compile(
    r"^\s*Tabla\s+(\d+(?:[.\-]\d+)*(?:\.(?-i:[a-zñ])\b|\s*\((?-i:[A-Za-zñ])\))?)(.*)",
    re.IGNORECASE,
)
_CONT_RE = re.compile(r"continuaci[oó]n", re.IGNORECASE)

_MIN_ROWS = 2
_MIN_COLS = 2

# Versión del extractor: participa del hash_region para que un cambio de lógica
# invalide el caché de extracciones previas (sin esto, una tabla mal extraída
# con la lógica vieja quedaría congelada para siempre por el cache-hit).
_EXTRACTION_VERSION = "v3"


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
                # "392.10 (A)" → "392.10(A)": id canónico sin espacios internos.
                "tabla_id": re.sub(r"\s+", "", m.group(1)),
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
# Celda "de datos": solo dígitos/puntuación numérica/rangos ("458", "65,4",
# "26-30", "1,5 – (30 Sd)" NO — contiene letras → no matchea, correcto: es dato
# pero con letras; el guard numérico solo sirve para descartar filas de datos
# como candidatas a subheader).
_NUMERIC_CELL_RE = re.compile(r"^[\d.,\s%–\-—()]+$")
_SUBHEADER_MAX_LEN = 30


def _is_subheader_row(header: List[str], row: List[str]) -> bool:
    """¿`row` es una fila de SUBCOLUMNAS del header multinivel?

    Señales exigidas (todas):
      - el nivel superior tiene celdas vacías (spans de un padre combinado);
      - `row` llena al menos una de esas columnas vacías (complementariedad);
      - las celdas llenas de `row` son etiquetas cortas (unidades tipo gr/m²,
        µm, kV) — ninguna es un valor numérico puro ni una oración.
    Una fila de datos ("Pletinas y láminas | 458 | 65,4 | …") tiene celdas
    numéricas → jamás se absorbe como subheader.
    """
    if not any(h == "" for h in header):
        return False
    filled = [c for c in row if c]
    if not filled:
        return False
    if any(_NUMERIC_CELL_RE.match(c) for c in filled):
        return False
    if not all(len(c) <= _SUBHEADER_MAX_LEN and not re.search(r"[.;:]\s", c) for c in filled):
        return False
    n = min(len(header), len(row))
    return any(header[j] == "" and row[j] for j in range(n))


def _ffill(row: List[str]) -> List[str]:
    """Colspan reconstruido: un padre combinado abarca las celdas vacías a su
    derecha ("PROMEDIO", "" → "PROMEDIO", "PROMEDIO")."""
    out: List[str] = []
    last = ""
    for c in row:
        if c:
            last = c
        out.append(last)
    return out


def merge_multilevel_headers(
    rows: List[List[str]], max_levels: int = 3
) -> Tuple[List[str], List[List[str]]]:
    """Devuelve (headers, body) fusionando headers de 2-3 niveles.

    "PROMEDIO" que abarca (gr/m², µm) produce los headers compuestos
    "PROMEDIO gr/m²" y "PROMEDIO µm" — un header por columna de datos, que es
    el contrato del JSON canónico. Un header plano pasa intacto (rows[0]).
    """
    if not rows:
        return [], []
    if len(rows) < 2:
        return rows[0], []

    levels = [rows[0]]
    idx = 1
    # rows[idx] puede absorberse como nivel de header mientras quede al menos
    # una fila de datos después (len(rows) - 1).
    while idx < len(rows) - 1 and len(levels) < max_levels and _is_subheader_row(levels[-1], rows[idx]):
        levels.append(rows[idx])
        idx += 1

    if len(levels) == 1:
        return rows[0], rows[1:]

    ncol = max(len(lv) for lv in levels)
    parents = [_ffill(lv) for lv in levels[:-1]]
    last = levels[-1]
    headers: List[str] = []
    for j in range(ncol):
        parts: List[str] = []
        for lv in parents:
            v = lv[j] if j < len(lv) else ""
            if v and (not parts or parts[-1] != v):
                parts.append(v)
        v = last[j] if j < len(last) else ""
        if v and (not parts or parts[-1] != v):
            parts.append(v)
        headers.append(" ".join(parts).strip())
    return headers, rows[idx:]


def _normalize_rows(data: List[List[Any]]) -> Tuple[List[str], List[List[str]]]:
    rows = [[("" if c is None else str(c).strip()) for c in row] for row in (data or [])]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return [], []
    # Ancho uniforme ANTES de fusionar niveles (pdfplumber puede devolver filas
    # de largos distintos).
    ncol = max(len(r) for r in rows)
    rows = [(r + [""] * (ncol - len(r)))[:ncol] for r in rows]
    headers, body = merge_multilevel_headers(rows)
    if not headers or not body:
        return [], []
    ncol = len(headers)
    body = [(r + [""] * (ncol - len(r)))[:ncol] for r in body]
    return headers, body


def _looks_like_prose(headers: List[str], rows: List[List[str]]) -> bool:
    """¿La "tabla" es texto corrido partido en columnas falsas?

    Firma real del bug (NTC 392.10): la estrategia de texto de pdfplumber cortó
    prosa a dos columnas en celdas que PARTEN PALABRAS —
    "(1) Conduct | ores individua | les. Debe perm | itirse la ins-".
    Señales: bordes de celda pegados (celda termina en letra y la siguiente
    empieza en minúscula) y celdas terminadas en guion de silabeo.
    """
    all_rows = [headers] + rows
    total = len(all_rows)
    if not total:
        return False
    glued_rows = 0
    hyphen_rows = 0
    for r in all_rows:
        if any(c.rstrip().endswith("-") for c in r if c):
            hyphen_rows += 1
        pairs = 0
        for a, b in zip(r, r[1:]):
            if a and b and a[-1].isalpha() and b[0].islower():
                pairs += 1
        if pairs >= 2:
            glued_rows += 1
    return (glued_rows / total) >= 0.4 or (hyphen_rows / total) >= 0.25


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
        # La estrategia de texto (sin rejilla vectorial) puede "ver" columnas en
        # prosa a dos columnas: si las celdas parten palabras, NO es una tabla —
        # se rechaza para que la cascada caiga a Vision/PNG en vez de indexar
        # ensalada de palabras como JSON canónico.
        if metodo == "text" and _looks_like_prose(headers, rows):
            log.info("[E3] estrategia text rechazada: la región parece prosa, no tabla")
            continue
        if _consistent(headers, rows):
            return {"metodo": metodo, "headers": headers, "rows": rows,
                    "bbox": [round(v, 1) for v in best.bbox]}
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Footnotes / línea "Fuente" (ruta geométrica; Vision los entrega vía notes)
# ──────────────────────────────────────────────────────────────────────────────
_FOOTNOTE_LINE_RE = re.compile(
    r"^\s*(?:\*+\s|\(\d+\)\s|\d\)\s|Nota[s]?\b|Fuente\b)", re.IGNORECASE
)


def collect_footnotes(pl_page, clip: Tuple[float, float, float, float]) -> Optional[str]:
    """Rescata footnotes y la línea 'Fuente: …' del texto de la región de la
    tabla. La extracción geométrica solo devuelve la rejilla; estas líneas
    viven fuera de ella y deben preservarse en notes (criterio de aceptación)."""
    try:
        text = pl_page.crop(clip).extract_text() or ""
    except Exception:
        return None
    keep = [ln.strip() for ln in text.splitlines() if _FOOTNOTE_LINE_RE.match(ln.strip())]
    return " · ".join(keep) if keep else None


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

    hash_region = hashlib.sha256(
        _EXTRACTION_VERSION.encode("utf-8") + b"".join(png_parts)
    ).hexdigest()[:16]

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
        # Footnotes y línea "Fuente" de la región (Vision ya los trae en notes;
        # la rejilla geométrica no los incluye y deben preservarse).
        notes = collect_footnotes(pdoc.pages[first_page - 1],
                                  (region.x0, region.y0, region.x1, region.y1))

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
