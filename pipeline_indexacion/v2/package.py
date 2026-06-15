"""Empaqueta el índice construido (out_dir) en una carpeta lista para subir a
MinIO MANUALMENTE — sin depender de la conexión inestable de Railway.

Produce:
  - Verificación de integridad de chroma_v2/ (DB + manifests + todos los PNG).
  - chroma_v2/SUBIR_A_MINIO.txt con instrucciones paso a paso (consola web + mc).
  - (opcional --zip) dist/chroma_v2_<version>.tar.gz como respaldo de un archivo.

El índice queda en pipeline_indexacion/chroma_v2/ y se sube a MinIO en el
bucket embeddings-store, en un prefijo NUEVO al lado del v1 (data/chroma_db/).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .config import V2Config

log = logging.getLogger(__name__)


def _dir_size(path: Path) -> Tuple[int, int]:
    """(bytes, n_archivos) de un árbol."""
    total = nfiles = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            nfiles += 1
    return total, nfiles


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n/1:.0f}{unit}" if False else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def verify(cfg: V2Config) -> Tuple[bool, List[str]]:
    """Verifica que chroma_v2/ esté completo y consistente."""
    out = cfg.out_dir
    problems: List[str] = []

    if not (out / "chroma.sqlite3").exists():
        problems.append("falta chroma.sqlite3 (¿corriste 'build'?)")
    shards = [d for d in out.iterdir() if d.is_dir() and d.name not in ("assets",)] if out.exists() else []
    if not shards:
        problems.append("falta el directorio de shard HNSW de Chroma")
    for f in ("index_manifest.json", "index_report.json", "assets_registry.sqlite"):
        if not (out / f).exists():
            problems.append(f"falta {f}")

    # Todos los PNG referenciados en el registro deben existir físicamente
    reg = out / "assets_registry.sqlite"
    if reg.exists():
        try:
            conn = sqlite3.connect(str(reg))
            conn.row_factory = sqlite3.Row
            missing = 0
            for table, col in (("tables_registry", "png_key"), ("figures_registry", "png_key")):
                for r in conn.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL"):
                    if not (out / "assets" / r[col]).exists():
                        missing += 1
            conn.close()
            if missing:
                problems.append(f"{missing} PNG referenciados en el registro no existen en assets/")
        except Exception as exc:
            problems.append(f"no se pudo leer el registro: {exc}")

    return (not problems), problems


_INSTRUCTIONS = """\
═══════════════════════════════════════════════════════════════════════════
  SUBIR ÍNDICE v2 A MINIO  (manual — sin depender del publish automático)
═══════════════════════════════════════════════════════════════════════════

Generado: {ts}
Bucket destino : {bucket}
Prefijo destino: {prefix}/        (al lado del v1, que está en {legacy}/)

CONTENIDO DE ESTA CARPETA ({this_dir}):
  - chroma.sqlite3 + carpeta <uuid>/  .... la BASE DE DATOS (lo que el bot LEE)
  - assets_registry.sqlite ............... registro de tablas/figuras (pequeño)
  - index_manifest.json / index_report.json  metadatos del índice
  - assets/tables/  · assets/figures/ .... imágenes (tablas y figuras)

TAMAÑOS:
  - Base de datos + registro + manifests : {db_size}  ({db_files} archivos)  ← CRÍTICO
  - assets/ (imágenes)                   : {assets_size}  ({assets_files} archivos)  ← pesado
  - TOTAL                                : {total_size}

──────────────────────────────────────────────────────────────────────────
 PLAN RECOMENDADO (sube en 2 pasos: primero lo crítico, luego lo pesado)
──────────────────────────────────────────────────────────────────────────

PASO 1 — La base de datos (hace que el bot funcione, sube en segundos):
  Sube estos archivos a  {bucket}/{prefix}/  :
    chroma.sqlite3
    <uuid>/            (la carpeta de shard completa)
    assets_registry.sqlite
    index_manifest.json
    index_report.json

PASO 2 — Las imágenes (las usa la futura herramienta de tablas; pueden esperar):
  Sube la carpeta  assets/  completa a  {bucket}/{prefix}/assets/

──────────────────────────────────────────────────────────────────────────
 OPCIÓN A — MinIO Client "mc"  (RECOMENDADO: reanudable, reintenta solo)
──────────────────────────────────────────────────────────────────────────
  1. Descarga mc.exe:  https://dl.min.io/client/mc/release/windows-amd64/mc.exe
  2. Configura el alias (una vez):
       mc alias set railway https://{endpoint} ACCESS_KEY SECRET_KEY
  3. Sube TODO con un comando (mc reintenta y reanuda lo que falle):
       mc mirror --overwrite "{this_dir}" railway/{bucket}/{prefix}
     (o en 2 pasos: primero sin assets, luego solo assets/)

──────────────────────────────────────────────────────────────────────────
 OPCIÓN B — Consola web de MinIO  (más simple, pero frágil con 628 archivos)
──────────────────────────────────────────────────────────────────────────
  1. Entra a la consola web de MinIO (Railway → servicio MinIO → URL).
  2. Bucket {bucket} → crea/entra a la "carpeta" {prefix}/.
  3. PASO 1: arrastra los archivos de la base de datos (lista de arriba).
  4. PASO 2: arrastra la carpeta assets/ (si corta, reintenta — sube solo lo que falte).

──────────────────────────────────────────────────────────────────────────
 ACTIVAR EL ÍNDICE v2 EN EL BOT  (cambio reversible al instante)
──────────────────────────────────────────────────────────────────────────
  En Railway, en el servicio del BOT, cambia la variable:
       MINIO_PREFIX={prefix}
  y redespliega. Al arrancar, bootstrap_sync descargará el índice v2.
  ROLLBACK: vuelve a poner  MINIO_PREFIX={legacy}  y redespliega (v1 intacto).

  Verifica en los logs del bot:  colección 'normativas': {chunks} items
═══════════════════════════════════════════════════════════════════════════
"""


def make_package(cfg: V2Config, *, target_prefix: str, make_zip: bool) -> None:
    ok, problems = verify(cfg)
    if not ok:
        log.error("[PACKAGE] La carpeta %s NO está completa:", cfg.out_dir)
        for p in problems:
            log.error("   ✗ %s", p)
        raise SystemExit("Corrige los problemas (normalmente: re-correr 'build') antes de empaquetar.")

    assets_dir = cfg.out_dir / "assets"
    a_bytes, a_files = _dir_size(assets_dir) if assets_dir.exists() else (0, 0)
    t_bytes, t_files = _dir_size(cfg.out_dir)
    db_bytes, db_files = t_bytes - a_bytes, t_files - a_files

    manifest = json.loads((cfg.out_dir / "index_manifest.json").read_text(encoding="utf-8"))
    import os
    endpoint = (os.getenv("MINIO_PUBLIC_ENDPOINT") or os.getenv("MINIO_ENDPOINT") or "<MINIO_ENDPOINT>")
    endpoint = endpoint.replace("https://", "").replace("http://", "")

    text = _INSTRUCTIONS.format(
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        bucket=cfg.bucket, prefix=target_prefix.strip("/"), legacy=cfg.legacy_prefix,
        this_dir=str(cfg.out_dir),
        db_size=_human(db_bytes), db_files=db_files,
        assets_size=_human(a_bytes), assets_files=a_files,
        total_size=_human(t_bytes),
        endpoint=endpoint, chunks=manifest.get("chunks", "?"),
    )
    instr_path = cfg.out_dir / "SUBIR_A_MINIO.txt"
    instr_path.write_text(text, encoding="utf-8")

    log.info("[PACKAGE] ✓ Índice verificado y completo en: %s", cfg.out_dir)
    log.info("[PACKAGE]   base de datos: %s (%d archivos)  ← crítico", _human(db_bytes), db_files)
    log.info("[PACKAGE]   imágenes:      %s (%d archivos)", _human(a_bytes), a_files)
    log.info("[PACKAGE]   TOTAL:         %s (%d archivos)", _human(t_bytes), t_files)
    log.info("[PACKAGE]   instrucciones: %s", instr_path)
    log.info("[PACKAGE]   destino MinIO:  %s/%s/  (al lado del v1 en %s/)",
             cfg.bucket, target_prefix.strip("/"), cfg.legacy_prefix)
    log.info("[PACKAGE]   → abre SUBIR_A_MINIO.txt para los pasos detallados")

    if make_zip:
        dist = cfg.work_dir.parent / "dist"
        dist.mkdir(parents=True, exist_ok=True)
        zname = dist / f"chroma_v2_{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
        log.info("[PACKAGE] Comprimiendo respaldo en %s (puede tardar) ...", zname)
        with tarfile.open(zname, "w:gz") as tar:
            tar.add(str(cfg.out_dir), arcname=target_prefix.strip("/"))
        log.info("[PACKAGE]   ✓ respaldo: %s (%s)", zname, _human(zname.stat().st_size))
        log.warning("[PACKAGE]   Nota: el .tar.gz es solo RESPALDO. MinIO NO lo descomprime; "
                    "para SERVIR el índice sube los archivos/carpetas, no el .tar.gz.")

    # Eco a consola tolerante a terminales no-UTF8 (Windows cp1252).
    try:
        print("\n" + text)
    except UnicodeEncodeError:
        import sys
        sys.stdout.buffer.write(("\n" + text).encode("utf-8", errors="replace"))
