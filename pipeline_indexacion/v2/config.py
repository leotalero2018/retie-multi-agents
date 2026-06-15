"""Configuración del pipeline v2 (rutas, umbrales, modelos)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]   # pipeline_indexacion/
REPO_ROOT = PIPELINE_ROOT.parent


def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class V2Config:
    # ── Rutas ────────────────────────────────────────────────────────────────
    docs_dir: Path = PIPELINE_ROOT / "docs"
    work_dir: Path = PIPELINE_ROOT / "v2_work"        # intermedios + cachés (persistente, NO se publica)
    out_dir: Path = PIPELINE_ROOT / "chroma_v2"       # índice final + assets + registro (se publica completo)

    # ── Índice ───────────────────────────────────────────────────────────────
    collection: str = field(default_factory=lambda: (
        (os.getenv("COLLECTION_NAMES", "normativas").split(",")[0].strip()) or "normativas"
    ))
    schema_version: int = 2

    # ── Chunking estructural ─────────────────────────────────────────────────
    chunk_tokens: int = int(os.getenv("V2_CHUNK_TOKENS", "450"))       # objetivo
    chunk_max_tokens: int = int(os.getenv("V2_CHUNK_MAX_TOKENS", "560"))  # tope duro por chunk

    # ── Layout / columnas ────────────────────────────────────────────────────
    fullwidth_ratio: float = 0.62      # bloque más ancho que esto = "full width" (corta bandas)
    boilerplate_freq: float = 0.35     # línea repetida en >35% de páginas = header/footer
    scanned_cover: float = 0.70        # imagen que cubre >70% de la página = escaneo
    bigimg_cover: float = 0.08         # imagen >8% del área = candidata a figura

    # ── Render ───────────────────────────────────────────────────────────────
    table_dpi: int = int(os.getenv("V2_TABLE_DPI", "300"))
    figure_dpi: int = int(os.getenv("V2_FIGURE_DPI", "200"))

    # ── Vision (GPT-4o) ──────────────────────────────────────────────────────
    vision_enabled: bool = field(default_factory=lambda: _env_bool("V2_VISION_ENABLED", "true"))
    vision_model: str = field(default_factory=lambda: os.getenv("VISION_MODEL", "gpt-4o"))
    vision_max_calls: int = int(os.getenv("V2_VISION_MAX_CALLS", "400"))   # presupuesto por corrida
    vision_caption_figures: bool = field(default_factory=lambda: _env_bool("V2_VISION_CAPTIONS", "true"))

    # ── Embeddings ───────────────────────────────────────────────────────────
    embed_batch: int = int(os.getenv("V2_EMBED_BATCH", "100"))

    # ── Publicación MinIO ────────────────────────────────────────────────────
    bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "embeddings-store"))
    legacy_prefix: str = field(default_factory=lambda: (
        os.getenv("MINIO_PREFIX", "data/chroma_db").strip().strip("/")
    ))
    versions_prefix: str = field(default_factory=lambda: (
        os.getenv("V2_VERSIONS_PREFIX", "chroma_db_versions").strip().strip("/")
    ))

    # ── Derivadas ────────────────────────────────────────────────────────────
    @property
    def inventory_dir(self) -> Path:
        return self.work_dir / "inventory"

    @property
    def text_dir(self) -> Path:
        return self.work_dir / "text"

    @property
    def structure_dir(self) -> Path:
        return self.work_dir / "structure"

    @property
    def assets_dir(self) -> Path:
        return self.work_dir / "assets"

    @property
    def registry_path(self) -> Path:
        return self.work_dir / "assets_registry.sqlite"

    @property
    def embcache_path(self) -> Path:
        return self.work_dir / "embcache.sqlite"

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.inventory_dir, self.text_dir, self.structure_dir,
                  self.assets_dir, self.assets_dir / "tables", self.assets_dir / "figures"):
            d.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """Carga .env raíz del repo y el .env local del pipeline (sin pisar valores)."""
    from dotenv import load_dotenv
    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
    local_env = PIPELINE_ROOT / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=False)
