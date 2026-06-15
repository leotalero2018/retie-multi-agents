"""Pipeline de indexación v2 — extracción multi-modal (texto + tablas + figuras).

Etapas (ver plan_indexacion_profesional.md):
  E0 profiler      — perfilado por página, inventario, incrementalidad
  E1 text_extract  — extracción layout-aware (columnas, headers/footers)
  E2 structure     — árbol normativo (artículos/secciones/numerales)
  E3 tables        — cascada de extracción de tablas + registro de activos
  E4 figures       — figuras/diagramas con captions
  E5 chunking      — chunking estructural con breadcrumbs
  E6 embed_cache   — caché SQLite de embeddings
  E7 validate      — reporte de calidad + gates
  E8 publish       — publicación atómica versionada en MinIO

CLI: python -m pipeline_indexacion.v2.run --help
"""
