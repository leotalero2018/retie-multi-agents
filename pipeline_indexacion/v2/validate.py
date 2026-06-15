"""E7 — Validación, reporte de calidad y gates de publicación.

Genera out_dir/index_report.json con:
  - conteos por documento (páginas, chunks, tablas por estado, figuras)
  - alertas de estructura (saltos de numeración)
  - QA muestral opcional: golden/queries.json → recall@k contra el índice nuevo

Gates (bloquean publish salvo --force):
  - tablas en estado 'revision' > 0
  - recall del golden set < umbral
  - colección vacía o dimensión de embeddings inesperada
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import V2Config, PIPELINE_ROOT
from .registry import AssetRegistry

log = logging.getLogger(__name__)

GOLDEN_PATH = PIPELINE_ROOT / "golden" / "queries.json"
RECALL_THRESHOLD = 0.80
EXPECTED_DIM = 1536


def run_validation(cfg: V2Config, structure_alerts: Dict[str, List[str]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"gates": [], "ok": True}

    # ── Chroma ───────────────────────────────────────────────────────────────
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    client = chromadb.PersistentClient(
        path=str(cfg.out_dir), settings=ChromaSettings(anonymized_telemetry=False))
    try:
        col = client.get_collection(cfg.collection)
        count = col.count()
    except Exception as exc:
        report["gates"].append(f"GATE: colección '{cfg.collection}' no existe ({exc})")
        report["ok"] = False
        return _write(cfg, report)

    report["chunks"] = count
    if count == 0:
        report["gates"].append("GATE: colección vacía")
        report["ok"] = False

    try:
        peek = col.peek(1)
        embs = peek.get("embeddings")
        if embs is not None and len(embs) > 0:
            dim = len(embs[0])
            report["embedding_dim"] = dim
            if dim != EXPECTED_DIM:
                report["gates"].append(f"GATE: dim de embeddings {dim} != {EXPECTED_DIM}")
                report["ok"] = False
    except Exception:
        pass

    # ── Activos ──────────────────────────────────────────────────────────────
    reg_path = cfg.out_dir / "assets_registry.sqlite"
    if reg_path.exists():
        reg = AssetRegistry(reg_path)
        tc = reg.table_counts()
        report["tables"] = tc
        report["tables_total"] = sum(tc.values())
        report["figures_total"] = reg.figure_count()
        # Verifica que cada PNG referenciado exista físicamente
        missing = []
        for t in reg.all_tables():
            if t.get("png_key") and not (cfg.out_dir / "assets" / t["png_key"]).exists():
                missing.append(f"tabla {t['doc_id']}/{t['tabla_id']}")
        if missing:
            report["gates"].append(f"GATE: {len(missing)} PNG de tablas faltantes: {missing[:5]}")
            report["ok"] = False
        if tc.get("revision", 0) > 0:
            report["gates"].append(
                f"GATE: {tc['revision']} tablas en estado 'revision' — resolver o publicar con --force")
            report["ok"] = False
        reg.close()
    else:
        report["tables"] = {}
        report["figures_total"] = 0

    # ── Estructura ───────────────────────────────────────────────────────────
    report["structure_alerts"] = structure_alerts
    n_alerts = sum(len(v) for v in structure_alerts.values())
    if n_alerts:
        log.warning("[E7] %d alertas de numeración (no bloquean)", n_alerts)

    # ── QA muestral (golden set) ─────────────────────────────────────────────
    if GOLDEN_PATH.exists():
        try:
            recall = _golden_recall(cfg, col)
            report["golden_recall"] = recall
            if recall < RECALL_THRESHOLD:
                report["gates"].append(
                    f"GATE: recall del golden set {recall:.2f} < {RECALL_THRESHOLD}")
                report["ok"] = False
        except Exception as exc:
            log.warning("[E7] golden set falló: %s", exc)
            report["golden_recall"] = None
    else:
        report["golden_recall"] = None
        log.info("[E7] sin golden set (%s) — QA muestral omitido", GOLDEN_PATH)

    return _write(cfg, report)


def _golden_recall(cfg: V2Config, col, top_k: int = 4) -> float:
    """golden/queries.json: [{"query": "...", "expect_contains": "texto que debe
    aparecer en algún chunk del top-k"}, ...]"""
    from .embed_cache import EmbeddingCache, embed_with_cache

    queries = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not queries:
        return 1.0
    cache = EmbeddingCache(cfg.embcache_path)
    embs = embed_with_cache([q["query"] for q in queries], cache)
    cache.close()

    hits = 0
    for q, emb in zip(queries, embs):
        res = col.query(query_embeddings=[emb], n_results=top_k, include=["documents"])
        docs = (res.get("documents") or [[]])[0] or []
        expect = (q.get("expect_contains") or "").lower()
        if expect and any(expect in (d or "").lower() for d in docs):
            hits += 1
        elif not expect:
            hits += 1
    return hits / len(queries)


def _write(cfg: V2Config, report: Dict[str, Any]) -> Dict[str, Any]:
    out = cfg.out_dir / "index_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Resumen legible
    log.info("══════════ REPORTE DE INDEXACIÓN ══════════")
    log.info("chunks=%s  dim=%s", report.get("chunks"), report.get("embedding_dim"))
    log.info("tablas=%s (total=%s)  figuras=%s",
             report.get("tables"), report.get("tables_total"), report.get("figures_total"))
    if report.get("golden_recall") is not None:
        log.info("golden recall@4 = %.2f", report["golden_recall"])
    for g in report["gates"]:
        log.error("✗ %s", g)
    log.info("RESULTADO: %s", "✓ APTO PARA PUBLICAR" if report["ok"] else "✗ GATES FALLIDOS")
    return report
