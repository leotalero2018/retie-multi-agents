"""E2 — Estructura jerárquica del documento normativo.

Segmenta el texto limpio (E1) en unidades normativas (artículo / sección /
numeral / anexo) con breadcrumb completo, p. ej.:
    "RETIE Libro 3 › CAPÍTULO 2 › ARTÍCULO 110 › 110.26"

Salida:
  - v2_work/structure/{doc_id}.toc.json   (tabla de contenido)
  - lista de segmentos {tipo, numero, titulo, breadcrumb, page_start,
    page_end, text} para el chunker (E5).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .config import V2Config

log = logging.getLogger(__name__)

# Patrones de encabezado, evaluados en orden. "context" actualiza el breadcrumb
# sin abrir segmento; "segment" abre una unidad nueva.
_PATTERNS: List[tuple[str, str, re.Pattern]] = [
    ("context", "titulo",   re.compile(r"^\s*T[IÍ]TULO\s+([IVXLCDM]+|\d+)\b[\s.:–-]*(.{0,120})", re.IGNORECASE)),
    ("context", "capitulo", re.compile(r"^\s*CAP[IÍ]TULO\s+([IVXLCDM]+|\d+)\b[\s.:–-]*(.{0,120})", re.IGNORECASE)),
    ("segment", "anexo",    re.compile(r"^\s*ANEXO\s+(GENERAL|[A-Z0-9]{1,4})\b[\s.:–-]*(.{0,120})", re.IGNORECASE)),
    ("segment", "articulo", re.compile(r"^\s*ART[IÍ]CULO\s+(\d{1,4})[°ºo]?\.?\s*[\s.:–-]*(.{0,120})", re.IGNORECASE)),
    ("segment", "seccion",  re.compile(r"^\s*SECCI[OÓ]N\s+(\d{1,4}(?:\.\d+)*)\b[\s.:–-]*(.{0,120})", re.IGNORECASE)),
    # Numerales tipo NTC/NEC: "220.55 Estufas eléctricas..." o "110-26 Espacios..."
    ("segment", "numeral",  re.compile(r"^\s*(\d{1,4}(?:[.\-]\d{1,3}){1,3})\s+([A-ZÁÉÍÓÚÑ(\"“'].{2,120})")),
]

_MIN_TITLE_LETTERS = 3  # un "título" debe tener letras (evita filas de tablas como falsos encabezados)


@dataclass
class Segment:
    tipo: str                 # articulo | seccion | numeral | anexo | preambulo
    numero: str
    titulo: str
    breadcrumb: str
    page_start: int
    page_end: int
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _letters(s: str) -> int:
    return len(re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", s))


def _match_heading(line: str):
    for kind, tipo, rx in _PATTERNS:
        m = rx.match(line)
        if not m:
            continue
        numero = (m.group(1) or "").strip().rstrip(".")
        titulo = (m.group(2) or "").strip()
        if tipo == "numeral":
            # exigir título con letras reales y descartar líneas tipo fila de tabla
            if _letters(titulo) < _MIN_TITLE_LETTERS:
                continue
            digits = len(re.findall(r"\d", line))
            if digits > len(line) * 0.4:
                continue
        return kind, tipo, numero, titulo
    return None


def segment_doc(cfg: V2Config, doc_meta: dict, pages: Dict[str, str]) -> List[Segment]:
    """Recorre el texto página a página y produce segmentos estructurales."""
    doc_id = doc_meta["doc_id"]
    doc_name = doc_meta.get("doc_name", doc_id)

    context: Dict[str, str] = {}  # titulo/capitulo vigentes
    segments: List[Segment] = []
    current = Segment("preambulo", "", "Preámbulo", doc_name, 1, 1)

    def _crumb(tipo: str, numero: str) -> str:
        parts = [doc_name]
        if context.get("titulo"):
            parts.append(f"Título {context['titulo']}")
        if context.get("capitulo"):
            parts.append(f"Capítulo {context['capitulo']}")
        label = {"articulo": "Art.", "seccion": "Sección", "numeral": "", "anexo": "Anexo"}[tipo]
        parts.append(f"{label} {numero}".strip())
        return " › ".join(parts)

    page_nums = sorted(int(k) for k in pages.keys())
    for pno in page_nums:
        for line in pages[str(pno)].splitlines():
            hit = _match_heading(line)
            if hit is None:
                current.lines.append(line)
                current.page_end = pno
                continue
            kind, tipo, numero, titulo = hit
            if kind == "context":
                context[tipo] = numero
                current.lines.append(line)
                current.page_end = pno
                continue
            # abrir nuevo segmento
            if current.text or current.tipo != "preambulo":
                segments.append(current)
            if tipo == "articulo":
                context.pop("ultimo_articulo", None)
                context["ultimo_articulo"] = numero
            current = Segment(
                tipo=tipo, numero=numero, titulo=titulo or numero,
                breadcrumb=_crumb(tipo, numero),
                page_start=pno, page_end=pno, lines=[line],
            )
    if current.text:
        segments.append(current)

    # Fallback: si el parser no encontró estructura, segmentar por página.
    structural = [s for s in segments if s.tipo != "preambulo"]
    if len(structural) < 3:
        log.warning("[E2] %s: estructura no detectada (%d encabezados) — fallback por página",
                    doc_id, len(structural))
        segments = []
        for pno in page_nums:
            txt = pages[str(pno)]
            if not txt.strip():
                continue
            seg = Segment("pagina", str(pno), f"Página {pno}", doc_name, pno, pno,
                          txt.splitlines())
            segments.append(seg)

    _write_toc(cfg, doc_id, segments)
    log.info("[E2] %s: %d segmentos (%s)", doc_id, len(segments),
             ", ".join(f"{t}={sum(1 for s in segments if s.tipo == t)}"
                       for t in sorted({s.tipo for s in segments})))
    return segments


def _write_toc(cfg: V2Config, doc_id: str, segments: List[Segment]) -> None:
    toc = [{
        "tipo": s.tipo, "numero": s.numero, "titulo": s.titulo[:120],
        "breadcrumb": s.breadcrumb, "page_start": s.page_start, "page_end": s.page_end,
        "chars": len(s.text),
    } for s in segments]
    (cfg.structure_dir / f"{doc_id}.toc.json").write_text(
        json.dumps(toc, ensure_ascii=False, indent=1), encoding="utf-8")


def numbering_alerts(segments: List[Segment]) -> List[str]:
    """Detecta saltos sospechosos en la numeración de artículos (QA del parser)."""
    alerts: List[str] = []
    nums = []
    for s in segments:
        if s.tipo == "articulo":
            try:
                nums.append((int(re.sub(r"\D", "", s.numero) or 0), s.numero, s.page_start))
            except ValueError:
                continue
    nums.sort(key=lambda x: x[2])
    for prev, cur in zip(nums, nums[1:]):
        if cur[0] > prev[0] + 3:  # salto de más de 3 artículos
            alerts.append(f"posible pérdida entre Art. {prev[1]} (p{prev[2]}) y Art. {cur[1]} (p{cur[2]})")
    return alerts
