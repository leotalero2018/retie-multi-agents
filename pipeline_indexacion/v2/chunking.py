"""E5 — Chunking estructural con breadcrumbs.

- Unidad de chunking = segmento normativo (E2). Si excede el presupuesto de
  tokens se parte en límites de literales/párrafos, NUNCA a mitad de frase.
- Cada chunk lleva su breadcrumb prepended ("[NTC 2050 › 220.55 — ...]"):
  mejora el retrieval y habilita citas exactas.
- Tablas y figuras se indexan como CHUNKS SINTÉTICOS (título + caption +
  primeras filas) que apuntan al registro de activos; el contenido completo
  se sirve siempre desde el JSON/PNG canónico, jamás desde el chunk.
- Metadata compatible con el runtime actual (doc_id, pagina, source...) +
  campos nuevos (articulo, tipo, schema_version=2).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import V2Config
from .registry import AssetRegistry
from .structure import Segment

log = logging.getLogger(__name__)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _ntokens(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:  # pragma: no cover
    def _ntokens(s: str) -> int:
        return max(1, len(s) // 4)

# Un literal "a) ..." o numeral "1. ..." abre una pieza nueva (frontera segura).
_PIECE_BOUNDARY = re.compile(r"^\s*(?:[a-zñ]\)|[A-ZÑ]\)|\d{1,2}\.\s|[•\-–]\s)")


@dataclass
class ChunkDoc:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def _split_pieces(text: str) -> List[str]:
    """Divide el texto del segmento en piezas indivisibles: párrafos y literales."""
    pieces: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                pieces.append("\n".join(current))
                current = []
            continue
        if _PIECE_BOUNDARY.match(line) and current:
            pieces.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def _split_oversize(piece: str, max_tokens: int) -> List[str]:
    """Última línea de defensa: una pieza sola mayor al tope se parte por
    oraciones (nunca a mitad de palabra)."""
    sentences = re.split(r"(?<=[.;:])\s+", piece)
    out: List[str] = []
    cur = ""
    for s in sentences:
        cand = f"{cur} {s}".strip() if cur else s
        if _ntokens(cand) <= max_tokens:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out or [piece]


def chunk_segments(cfg: V2Config, doc_meta: dict, segments: List[Segment]) -> List[ChunkDoc]:
    doc_id = doc_meta["doc_id"]
    doc_name = doc_meta.get("doc_name", doc_id)
    chunks: List[ChunkDoc] = []
    counter = 0

    base_meta = {
        "schema_version": cfg.schema_version,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "source": doc_name,
        "doc_tipo": doc_meta.get("doc_tipo", ""),
        "vigente": bool(doc_meta.get("vigente", True)),
        "entidad": doc_meta.get("entidad", ""),
        "version": doc_meta.get("version", ""),
    }

    for seg in segments:
        if not seg.text:
            continue
        header = f"[{seg.breadcrumb}]"
        if seg.titulo and seg.titulo not in seg.breadcrumb:
            header = f"[{seg.breadcrumb} — {seg.titulo[:80]}]"
        header_tokens = _ntokens(header)
        budget = max(80, cfg.chunk_tokens - header_tokens)
        hard_max = max(120, cfg.chunk_max_tokens - header_tokens)

        pieces: List[str] = []
        for p in _split_pieces(seg.text):
            if _ntokens(p) > hard_max:
                pieces.extend(_split_oversize(p, hard_max))
            else:
                pieces.append(p)

        groups: List[str] = []
        cur = ""
        for p in pieces:
            cand = f"{cur}\n\n{p}".strip() if cur else p
            if _ntokens(cand) <= budget:
                cur = cand
            else:
                if cur:
                    groups.append(cur)
                cur = p
        if cur:
            groups.append(cur)

        for body in groups:
            counter += 1
            meta = {
                **base_meta,
                "pagina": seg.page_start,
                "page": seg.page_start,
                "page_end": seg.page_end,
                "tipo": "texto",
                "articulo": f"{seg.tipo} {seg.numero}".strip() if seg.tipo != "preambulo" else "",
                "breadcrumb": seg.breadcrumb,
                "chunk_id": counter,
            }
            chunks.append(ChunkDoc(
                chunk_id=f"{doc_id}_c{counter:05d}",
                text=f"{header}\n{body}",
                metadata=meta,
            ))

    log.info("[E5] %s: %d chunks de texto", doc_id, len(chunks))
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Chunks sintéticos de activos (tablas / figuras)
# ──────────────────────────────────────────────────────────────────────────────
_MAX_PREVIEW_ROWS = 3


def asset_chunks(cfg: V2Config, doc_meta: dict, registry: AssetRegistry) -> List[ChunkDoc]:
    doc_id = doc_meta["doc_id"]
    doc_name = doc_meta.get("doc_name", doc_id)
    out: List[ChunkDoc] = []

    for t in registry.all_tables(doc_id):
        parts = [f"[{doc_name} — Tabla {t['tabla_id']}] {t.get('titulo') or ''}".strip()]
        if t.get("articulo_padre"):
            parts.append(f"Pertenece a: {t['articulo_padre']}")
        if t.get("json_key"):
            try:
                data = json.loads((cfg.assets_dir / t["json_key"]).read_text(encoding="utf-8"))
                headers = data.get("headers") or []
                rows = (data.get("rows") or [])[:_MAX_PREVIEW_ROWS]
                if headers:
                    parts.append("Columnas: " + " | ".join(str(h) for h in headers))
                for r in rows:
                    parts.append("Fila: " + " | ".join(str(c) for c in r))
            except Exception:
                pass
        out.append(ChunkDoc(
            chunk_id=f"{doc_id}_tabla_{re.sub(r'[^0-9A-Za-z.()-]', '_', t['tabla_id'])}",
            text="\n".join(parts),
            metadata={
                "schema_version": cfg.schema_version,
                "doc_id": doc_id, "doc_name": doc_name, "source": doc_name,
                "vigente": bool(doc_meta.get("vigente", True)),
                "pagina": t.get("pagina_inicio") or 0, "page": t.get("pagina_inicio") or 0,
                "tipo": "tabla", "tabla_id": t["tabla_id"],
                "estado": t.get("estado", ""), "png_key": t.get("png_key", ""),
                "chunk_id": 0,
            },
        ))

    for f in registry.all_figures(doc_id):
        out.append(ChunkDoc(
            chunk_id=f"{doc_id}_fig_{re.sub(r'[^0-9A-Za-z.()_-]', '_', f['figura_id'])}",
            text=f"[{doc_name} — Figura {f['figura_id']}]\n{f.get('caption') or ''}",
            metadata={
                "schema_version": cfg.schema_version,
                "doc_id": doc_id, "doc_name": doc_name, "source": doc_name,
                "vigente": bool(doc_meta.get("vigente", True)),
                "pagina": f.get("pagina") or 0, "page": f.get("pagina") or 0,
                "tipo": "figura", "figura_id": f["figura_id"],
                "png_key": f.get("png_key", ""),
                "chunk_id": 0,
            },
        ))

    if out:
        log.info("[E5] %s: +%d chunks sintéticos de activos", doc_id, len(out))
    return out
