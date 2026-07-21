# retie_agent/services/table_assets.py
"""Acceso runtime al registro canónico de tablas del índice v2.

El pipeline v2 extrae cada tabla UNA vez (pdfplumber/Vision) y publica junto al
índice Chroma: `assets_registry.sqlite` + `assets/tables/<doc>/<tabla>.json`
(estructura canónica {title, headers, rows, notes}) + `<tabla>.png` (recorte
del PDF, fidelidad 100 %). Los chunks del índice solo llevan un PREVIEW de 3
filas por diseño — el contenido completo SIEMPRE debe servirse desde aquí.

Este módulo era la mitad que faltaba: el runtime reconstruía las tablas con un
LLM desde texto ambiguo en vez de leer el JSON canónico. `lookup_table(ref)`
resuelve una referencia del usuario ("2.3.26.2.2.1.a", "392.10(A)", "220.55")
al activo canónico. Nunca lanza: sin registro / sin match → None y el llamador
degrada al camino LLM.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from retie_agent.config import settings

logger = logging.getLogger(__name__)

# Confianza por estado de extracción (registry del pipeline v2):
#   verificada > extraida > solo_imagen (solo PNG) > revision (no confiable)
_ESTADO_RANK = {"verificada": 3, "extraida": 2, "solo_imagen": 1, "revision": 0}


@dataclass
class CanonicalTable:
    doc_id: str
    tabla_id: str
    titulo: str
    estado: str
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    notes: Optional[str] = None
    markdown: Optional[str] = None
    png_path: Optional[Path] = None
    pagina: Optional[int] = None
    # Cómo matcheó la referencia del usuario contra el id registrado:
    #   exact              → mismo id (normalizado)
    #   stored_extends_ref → el registro es más específico ("392.10" pide,
    #                        "392.10(A)" existe) — confiable
    #   ref_extends_stored → la referencia es más específica que el registro
    #                        ("2.3.26.2.2.1.a" pide, "2.3.26.2.2.1" existe):
    #                        firma de un índice construido ANTES del fix de IDs
    #                        truncados — su JSON puede ser OTRA tabla (las .a/.b
    #                        se fusionaban). NO servirlo como estructura.
    match: str = "exact"


def _root() -> Path:
    """Raíz del índice sincronizado (bootstrap_sync baja assets junto a Chroma)."""
    return Path(getattr(settings, "CHROMA_PERSIST_DIR", "data/chroma_db"))


def _registry_path() -> Path:
    return _root() / "assets_registry.sqlite"


def _norm(ref: str) -> str:
    """Clave de comparación: sin espacios, minúsculas ("392.10 (A)" ≡ "392.10(a)")."""
    return re.sub(r"\s+", "", str(ref or "")).lower()


def _load_json(json_key: Optional[str]) -> Dict[str, Any]:
    if not json_key:
        return {}
    try:
        path = _root() / "assets" / json_key
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — degradar, nunca romper la respuesta
        logger.warning("table_assets: no pude leer JSON canónico %s: %s", json_key, exc)
        return {}


def lookup_table(ref: str) -> Optional[CanonicalTable]:
    """Resuelve una referencia de tabla al activo canónico. Nunca lanza.

    Matching (mejor a peor):
      1. igualdad normalizada ("392.10(A)" ≡ "392.10 (a)");
      2. el id registrado extiende la referencia ("392.10" pide, "392.10(A)" existe);
      3. la referencia extiende el id registrado ("2.3.26.2.2.1.a" pide,
         "2.3.26.2.2.1" quedó registrado por el truncamiento histórico).
    Empates: mejor `estado`, luego id más largo (más específico).
    """
    key = _norm(ref)
    if not key:
        return None
    reg = _registry_path()
    if not reg.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{reg.as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT doc_id, tabla_id, titulo, estado, json_key, png_key, "
                "markdown, pagina_inicio FROM tables_registry"
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("table_assets: registro ilegible (%s): %s", reg, exc)
        return None

    best = None
    best_match = "exact"
    best_score = None
    for r in rows:
        tid = _norm(r["tabla_id"])
        if tid == key:
            match, rank = "exact", 3
        elif tid.startswith(key):
            match, rank = "stored_extends_ref", 2
        elif key.startswith(tid):
            match, rank = "ref_extends_stored", 1
        else:
            continue
        score = (rank, _ESTADO_RANK.get(r["estado"], 0), len(tid))
        if best_score is None or score > best_score:
            best, best_match, best_score = r, match, score
    if best is None:
        return None

    data = _load_json(best["json_key"])
    png_path: Optional[Path] = None
    if best["png_key"]:
        candidate = _root() / "assets" / best["png_key"]
        if candidate.exists():
            png_path = candidate

    headers = [str(h) for h in (data.get("headers") or [])]
    rows_out = [[str(c) for c in row] for row in (data.get("rows") or [])]
    return CanonicalTable(
        doc_id=best["doc_id"],
        tabla_id=best["tabla_id"],
        titulo=(data.get("title") or best["titulo"] or f"Tabla {best['tabla_id']}").strip(),
        estado=best["estado"] or "",
        headers=headers,
        rows=rows_out,
        notes=(data.get("notes") or None),
        markdown=best["markdown"],
        png_path=png_path,
        pagina=best["pagina_inicio"],
        match=best_match,
    )
