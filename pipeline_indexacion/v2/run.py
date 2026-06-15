"""Orquestador CLI del pipeline de indexación v2.

Uso:
  python -m pipeline_indexacion.v2.run profile                 # E0 solo perfilado
  python -m pipeline_indexacion.v2.run build                   # E0→E7 (sin publicar)
  python -m pipeline_indexacion.v2.run build --doc retie_libro3
  python -m pipeline_indexacion.v2.run build --no-vision       # sin GPT-4o Vision
  python -m pipeline_indexacion.v2.run validate                # E7 sobre el índice ya construido
  python -m pipeline_indexacion.v2.run publish                 # E8 (respeta gates)
  python -m pipeline_indexacion.v2.run publish --force
  python -m pipeline_indexacion.v2.run all                     # build + publish
  python -m pipeline_indexacion.v2.run versions                # versiones publicadas

El registro de documentos (DOCUMENT_REGISTRY) se reutiliza de
pipeline_indexacion/indexar_normativas.py — una sola fuente de verdad.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Permite ejecutar también como script directo
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline_indexacion.v2.config import V2Config, load_env  # noqa: E402

log = logging.getLogger("v2")


def _force_utf8_stdio() -> None:
    """Evita UnicodeEncodeError en consolas Windows (cp1252) al imprimir
    acentos / caracteres de caja en logs, --help y reportes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except Exception:
            pass


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _docs(only: str | None) -> List[dict]:
    from pipeline_indexacion.indexar_normativas import DOCUMENT_REGISTRY, _find_pdf
    out = []
    for d in DOCUMENT_REGISTRY:
        if only and d["doc_id"] != only:
            continue
        pdf = _find_pdf(d["filename"])
        if pdf is None:
            log.warning("[SKIP] PDF no encontrado: %s", d["filename"])
            continue
        out.append({**d, "_pdf_path": pdf})
    if only and not out:
        raise SystemExit(f"doc_id '{only}' no existe en DOCUMENT_REGISTRY o falta su PDF")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Etapas
# ──────────────────────────────────────────────────────────────────────────────
def cmd_profile(cfg: V2Config, args) -> None:
    from pipeline_indexacion.v2.profiler import profile_doc
    for d in _docs(args.doc):
        profile_doc(cfg, d, d["_pdf_path"])


def cmd_build(cfg: V2Config, args) -> None:
    from pipeline_indexacion.v2.profiler import profile_doc
    from pipeline_indexacion.v2.text_extract import extract_doc_text
    from pipeline_indexacion.v2.structure import segment_doc, numbering_alerts
    from pipeline_indexacion.v2.tables import extract_tables_for_doc
    from pipeline_indexacion.v2.figures import extract_figures_for_doc
    from pipeline_indexacion.v2.chunking import chunk_segments, asset_chunks
    from pipeline_indexacion.v2.index_build import build_index
    from pipeline_indexacion.v2.registry import AssetRegistry
    from pipeline_indexacion.v2.vision import VisionBudget
    from pipeline_indexacion.v2.validate import run_validation

    if args.no_vision:
        cfg.vision_enabled = False

    t0 = time.time()
    registry = AssetRegistry(cfg.registry_path)
    budget = VisionBudget(cfg.vision_max_calls)
    all_chunks = []
    doc_summaries: Dict[str, dict] = {}
    structure_alerts: Dict[str, List[str]] = {}

    for d in _docs(args.doc):
        doc_id = d["doc_id"]
        pdf = d["_pdf_path"]
        log.info("════════ %s (%s) ════════", doc_id, pdf.name)

        inv = profile_doc(cfg, d, pdf)                                   # E0
        pages = extract_doc_text(cfg, d, pdf, inv, force=args.force_text)  # E1
        segments = segment_doc(cfg, d, pages)                            # E2
        structure_alerts[doc_id] = numbering_alerts(segments)

        t_stats = {}
        if not args.skip_tables:
            t_stats = extract_tables_for_doc(cfg, d, pdf, inv, pages,    # E3
                                             registry, budget, segments)
        n_figs = 0
        if not args.skip_figures:
            n_figs = extract_figures_for_doc(cfg, d, pdf, inv, pages,    # E4
                                             registry, budget)

        chunks = chunk_segments(cfg, d, segments)                        # E5
        chunks += asset_chunks(cfg, d, registry)
        all_chunks.extend(chunks)

        doc_summaries[doc_id] = {
            "filename": pdf.name, "pages": inv["n_pages"],
            "chunks": len(chunks), "tables": t_stats, "figures": n_figs,
            "vigente": d.get("vigente", True),
        }

    registry.close()
    if not all_chunks:
        raise SystemExit("La construcción no produjo chunks — revisa los logs")

    build_index(cfg, all_chunks, doc_summaries)                          # E6 + índice
    log.info("[BUILD] %d chunks totales en %.1f min (vision usadas: %d)",
             len(all_chunks), (time.time() - t0) / 60, budget.used)

    run_validation(cfg, structure_alerts)                                # E7


def cmd_validate(cfg: V2Config, args) -> None:
    from pipeline_indexacion.v2.validate import run_validation
    run_validation(cfg, {})


def cmd_publish(cfg: V2Config, args) -> None:
    from pipeline_indexacion.v2.publish import publish
    publish(cfg, force=args.force, mirror_legacy=not args.no_mirror,
            version=getattr(args, "version", None))


def cmd_package(cfg: V2Config, args) -> None:
    from pipeline_indexacion.v2.package import make_package
    make_package(cfg, target_prefix=args.target_prefix, make_zip=args.zip)


def cmd_versions(cfg: V2Config, args) -> None:
    from pipeline_indexacion.v2.publish import list_versions
    for v in list_versions(cfg):
        print(v)


def cmd_all(cfg: V2Config, args) -> None:
    cmd_build(cfg, args)
    cmd_publish(cfg, args)


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _force_utf8_stdio()
    _setup_logging()
    load_env()

    parser = argparse.ArgumentParser(
        prog="pipeline_indexacion.v2",
        description="Pipeline de indexación multi-modal (texto + tablas + figuras)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--doc", help="procesar solo este doc_id (p. ej. retie_libro3)")
    common.add_argument("--no-vision", action="store_true",
                        help="desactivar GPT-4o Vision (tablas escaneadas quedan solo_imagen)")
    common.add_argument("--skip-tables", action="store_true", help="omitir E3")
    common.add_argument("--skip-figures", action="store_true", help="omitir E4")
    common.add_argument("--force-text", action="store_true",
                        help="re-extraer texto aunque el PDF no haya cambiado")
    common.add_argument("--force", action="store_true",
                        help="publicar aunque haya gates fallidos")
    common.add_argument("--no-mirror", action="store_true",
                        help="no espejar al prefijo legacy (el bot no verá esta versión)")
    common.add_argument("--version", default=None,
                        help="reanudar una publicación interrumpida (p. ej. v20260612-201021)")
    common.add_argument("--target-prefix", default="chroma_v2",
                        help="prefijo destino en MinIO para 'package' (default: chroma_v2)")
    common.add_argument("--zip", action="store_true",
                        help="además crear un .tar.gz de respaldo en dist/ ('package')")

    sub.add_parser("profile", parents=[common], help="E0: perfilado e inventario")
    sub.add_parser("build", parents=[common], help="E0-E7: construir indice completo")
    sub.add_parser("validate", parents=[common], help="E7: reporte + gates")
    sub.add_parser("package", parents=[common],
                   help="verificar chroma_v2/ + instrucciones para subir a MinIO manualmente")
    sub.add_parser("publish", parents=[common], help="E8: publicar a MinIO (automático)")
    sub.add_parser("all", parents=[common], help="build + publish")
    sub.add_parser("versions", parents=[common], help="listar versiones publicadas")

    args = parser.parse_args()
    cfg = V2Config()
    cfg.ensure_dirs()

    {"profile": cmd_profile, "build": cmd_build, "validate": cmd_validate,
     "package": cmd_package, "publish": cmd_publish, "versions": cmd_versions,
     "all": cmd_all}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
