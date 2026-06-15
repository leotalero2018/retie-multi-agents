"""Registro de activos (tablas y figuras) — SQLite.

Es la "base de datos" consultable de forma EXACTA en runtime:
    tabla_id "220.55" → doc, páginas, bbox, PNG, JSON, estado.

El archivo se publica JUNTO al índice Chroma (mismo prefijo versionado en
MinIO), de modo que índice y activos siempre son consistentes entre sí.

Estados de una tabla:
    verificada   extracción geométrica y visión coinciden (o validador dedicado OK)
    extraida     una estrategia produjo JSON estructurado consistente
    solo_imagen  no se pudo estructurar: se sirve el PNG original (fidelidad 100%)
    revision     extracciones contradictorias → requiere revisión humana
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tables_registry (
    doc_id        TEXT NOT NULL,
    tabla_id      TEXT NOT NULL,
    titulo        TEXT,
    pagina_inicio INTEGER,
    pagina_fin    INTEGER,
    bbox          TEXT,            -- JSON [x0,y0,x1,y1] (página inicio)
    png_key       TEXT,            -- ruta relativa en assets/ (también clave MinIO)
    json_key      TEXT,            -- ruta relativa del JSON canónico (si se extrajo)
    markdown      TEXT,            -- representación entregable en chat
    estado        TEXT NOT NULL,   -- verificada | extraida | solo_imagen | revision
    metodo        TEXT,            -- lattice | text | vision | none
    articulo_padre TEXT,
    hash_region   TEXT,            -- sha256 del PNG → caché de Vision entre corridas
    n_filas       INTEGER DEFAULT 0,
    n_cols        INTEGER DEFAULT 0,
    telegram_file_id TEXT,         -- caché de reenvío (lo llena el bot en runtime)
    PRIMARY KEY (doc_id, tabla_id)
);
CREATE TABLE IF NOT EXISTS figures_registry (
    doc_id        TEXT NOT NULL,
    figura_id     TEXT NOT NULL,
    caption       TEXT,
    pagina        INTEGER,
    bbox          TEXT,
    png_key       TEXT,
    hash_region   TEXT,
    telegram_file_id TEXT,
    PRIMARY KEY (doc_id, figura_id)
);
CREATE INDEX IF NOT EXISTS idx_tables_estado ON tables_registry(estado);
"""


class AssetRegistry:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # ── tablas ────────────────────────────────────────────────────────────────
    def upsert_table(self, row: Dict[str, Any]) -> None:
        cols = ("doc_id", "tabla_id", "titulo", "pagina_inicio", "pagina_fin", "bbox",
                "png_key", "json_key", "markdown", "estado", "metodo",
                "articulo_padre", "hash_region", "n_filas", "n_cols")
        values = [row.get(c) for c in cols]
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO tables_registry ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
                f"ON CONFLICT(doc_id, tabla_id) DO UPDATE SET "
                + ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("doc_id", "tabla_id")),
                values,
            )

    def get_table(self, doc_id: str, tabla_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM tables_registry WHERE doc_id=? AND tabla_id=?", (doc_id, tabla_id))
        r = cur.fetchone()
        return dict(r) if r else None

    def find_table_any_doc(self, tabla_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM tables_registry WHERE tabla_id=? ORDER BY doc_id", (tabla_id,))
        return [dict(r) for r in cur.fetchall()]

    def all_tables(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if doc_id:
            cur = self._conn.execute(
                "SELECT * FROM tables_registry WHERE doc_id=? ORDER BY pagina_inicio", (doc_id,))
        else:
            cur = self._conn.execute(
                "SELECT * FROM tables_registry ORDER BY doc_id, pagina_inicio")
        return [dict(r) for r in cur.fetchall()]

    def cached_extraction(self, doc_id: str, tabla_id: str, hash_region: str) -> Optional[Dict[str, Any]]:
        """Si la región no cambió desde la corrida anterior y ya hay JSON,
        reutiliza la extracción (evita re-pagar Vision)."""
        row = self.get_table(doc_id, tabla_id)
        if row and row.get("hash_region") == hash_region and row.get("json_key"):
            return row
        return None

    def table_counts(self) -> Dict[str, int]:
        cur = self._conn.execute(
            "SELECT estado, COUNT(*) AS n FROM tables_registry GROUP BY estado")
        return {r["estado"]: r["n"] for r in cur.fetchall()}

    # ── figuras ───────────────────────────────────────────────────────────────
    def upsert_figure(self, row: Dict[str, Any]) -> None:
        cols = ("doc_id", "figura_id", "caption", "pagina", "bbox", "png_key", "hash_region")
        values = [row.get(c) for c in cols]
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO figures_registry ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
                f"ON CONFLICT(doc_id, figura_id) DO UPDATE SET "
                + ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("doc_id", "figura_id")),
                values,
            )

    def get_figure(self, doc_id: str, figura_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM figures_registry WHERE doc_id=? AND figura_id=?", (doc_id, figura_id))
        r = cur.fetchone()
        return dict(r) if r else None

    def all_figures(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if doc_id:
            cur = self._conn.execute(
                "SELECT * FROM figures_registry WHERE doc_id=? ORDER BY pagina", (doc_id,))
        else:
            cur = self._conn.execute("SELECT * FROM figures_registry ORDER BY doc_id, pagina")
        return [dict(r) for r in cur.fetchall()]

    def figure_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM figures_registry").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def table_json_payload(titulo: Optional[str], headers: List[str], rows: List[List[Any]],
                       notes: Optional[str] = None) -> str:
    return json.dumps(
        {"title": titulo, "headers": headers, "rows": rows, "notes": notes},
        ensure_ascii=False,
    )
