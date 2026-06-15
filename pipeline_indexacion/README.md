# Pipeline de Indexación de Normativas Eléctricas — v2 (multi-modal)

Indexa los PDFs de normativas eléctricas colombianas en ChromaDB con extracción **multi-modal**: texto layout-aware, **tablas** (incluidas las que existen solo como imagen en PDFs escaneados), **figuras/diagramas**, estructura normativa (artículos/numerales) y publicación atómica versionada en MinIO.

> El pipeline anterior (`indexar_normativas.py`) queda como legacy: extraía la NTC 2050 entrelazando sus dos columnas y no veía tablas-imagen ni figuras. **Usa siempre el v2** (`pipeline_indexacion/v2/`). El `DOCUMENT_REGISTRY` (lista de documentos) sigue viviendo en `indexar_normativas.py` y es compartido por ambos.

---

## Qué hace el v2 que el v1 no hacía

| Capacidad | v1 (legacy) | v2 |
|---|---|---|
| NTC 2050 a 2 columnas | ❌ entrelazaba columnas a mitad de oración | ✅ extracción por columna en orden de lectura |
| Headers/footers repetidos | ❌ contaminaban cada chunk | ✅ detección y eliminación automática |
| Tablas nativas | ❌ texto plano revuelto | ✅ extracción geométrica → JSON estructurado |
| Tablas como imagen (RETIE escaneado) | ❌ invisibles | ✅ GPT-4o Vision → JSON + **PNG del recorte exacto** |
| Figuras/diagramas | ❌ invisibles | ✅ PNG + caption por Vision, consultables |
| Ubicación exacta | solo página | ✅ doc + páginas + **bbox** por tabla/figura |
| Chunking | 800 caracteres ciegos | ✅ estructural por artículo/numeral con breadcrumb |
| Incrementalidad | re-procesaba y re-pagaba todo | ✅ skip por hash de archivo + caché de embeddings + caché de Vision |
| Validación | ninguna | ✅ reporte de calidad + gates antes de publicar |
| Publicación | borrar y subir (no atómica) | ✅ versionada con `latest.json` + espejo legacy para el bot |

---

## Documentos indexados

Todos en la **colección única `normativas`**; la vigencia queda en la metadata `vigente` de cada chunk. Se definen en `DOCUMENT_REGISTRY` (`indexar_normativas.py`).

| doc_id | Archivo | Tipo de PDF | vigente |
|---|---|---|---|
| `ntc2050_v2` | NTC 2050 V2 - Código Eléctrico Colombiano | nativo, 2 columnas, 1.124 págs | ✅ |
| `ntc2050_erratas` | NTC 2050 - Fe de Erratas | nativo | ✅ |
| `retie_libro1` | RETIE Libro 1 - Disposiciones Generales | **escaneado** (43 págs) | — |
| `retie_libro2` | RETIE Libro 2 - Productos | **escaneado** (105 págs) | — |
| `retie_libro3` | RETIE Libro 3 - Instalaciones | **escaneado** (168 págs) | — |
| `retie_libro4` | RETIE Libro 4 - Evaluación de la Conformidad | **escaneado** (44 págs) | — |

Los PDFs deben estar en `pipeline_indexacion/docs/`.

---

## Requisitos previos

### 1. Entorno

Desde la raíz del repo, con el venv del proyecto activo. Dependencias clave (ya en `requirements.txt`): `chromadb`, `PyMuPDF`, `pdfplumber`, `tiktoken`, `minio`, `openai`, `python-dotenv`.

### 2. Variables en `.env` (raíz del repo)

```env
# Embeddings y Vision (obligatorio)
OPENAI_API_KEY=sk-...

# MinIO (solo para publish)
MINIO_PUBLIC_ENDPOINT=<host:puerto>      # ej: bucket.railway.app:443
MINIO_ROOT_USER=<access_key>
MINIO_ROOT_PASSWORD=<secret_key>
MINIO_BUCKET_NAME=embeddings-store
MINIO_PREFIX=data/chroma_db              # prefijo que consume el bot (espejo legacy)
```

Opcionales del v2 (defaults sensatos):

```env
V2_VISION_ENABLED=true        # GPT-4o Vision para tablas escaneadas/rotas y captions
V2_VISION_MAX_CALLS=400       # presupuesto de llamadas Vision por corrida
VISION_MODEL=gpt-4o
V2_CHUNK_TOKENS=450           # tamaño objetivo de chunk (tokens)
V2_TABLE_DPI=300              # resolución del PNG de tablas
```

> **Costo de Vision:** cada región se paga UNA sola vez (caché por hash en el registro). La corrida inicial completa del corpus usa ~200-350 llamadas ≈ USD 3-8. Las siguientes corridas reutilizan todo lo no cambiado.

---

## Indexación — paso a paso

Todos los comandos se ejecutan **desde la raíz del repo**.

### Paso 0 — Coloca/actualiza los PDFs

Copia los PDFs a `pipeline_indexacion/docs/`. Si agregas un documento **nuevo**, regístralo primero en `DOCUMENT_REGISTRY` (`indexar_normativas.py`) con su `doc_id`, `filename`, `doc_name`, `vigente` y `alias`.

### Paso 1 — Perfilado (opcional pero recomendado la primera vez)

```bash
python -m pipeline_indexacion.v2.run profile
```

Muestra por documento: páginas nativas/escaneadas, páginas a 2 columnas y páginas con tablas. Sirve para anticipar cuántas llamadas Vision necesitará el build. El inventario queda en `v2_work/inventory/`.

### Paso 2 — Build (construye el índice completo, NO publica)

```bash
python -m pipeline_indexacion.v2.run build
```

Ejecuta E0→E7 para todos los documentos:

```
E0 perfilado        → skip automático de PDFs sin cambios (hash)
E1 texto            → columnas en orden de lectura + headers/footers eliminados
E2 estructura       → artículos/numerales/secciones (toc.json por documento)
E3 tablas           → cascada: geometría → Vision → JSON + PNG del recorte + registro
E4 figuras          → PNG + caption + registro (filtra logos/marcas repetidas)
E5 chunking         → estructural con breadcrumbs + chunks sintéticos de tablas/figuras
E6 embeddings       → text-embedding-3-small con caché SQLite (solo paga lo nuevo)
E7 validación       → index_report.json + gates
```

Variantes útiles:

```bash
python -m pipeline_indexacion.v2.run build --doc retie_libro3   # un solo documento
python -m pipeline_indexacion.v2.run build --no-vision          # sin GPT-4o (tablas escaneadas quedan solo_imagen)
python -m pipeline_indexacion.v2.run build --skip-figures       # omitir figuras
python -m pipeline_indexacion.v2.run build --force-text         # re-extraer texto aunque el PDF no cambió
```

**Tiempo estimado:** corrida inicial completa 30-60 min (Vision es lo lento); re-corridas con caché caliente < 5 min.

### Paso 3 — Revisa el reporte y la cola de revisión

El build termina imprimiendo el reporte (también en `chroma_v2/index_report.json`):

```
chunks=XXXX  dim=1536
tablas={'verificada': N, 'extraida': N, 'solo_imagen': N, 'revision': N}  figuras=NN
golden recall@4 = 0.95
RESULTADO: ✓ APTO PARA PUBLICAR
```

Estados de tabla:
- **verificada** — geometría y Vision coinciden: máxima confianza.
- **extraida** — una estrategia produjo JSON consistente.
- **solo_imagen** — no se pudo estructurar: en runtime se sirve el **PNG original del recorte** (fidelidad perfecta garantizada).
- **revision** — extracciones contradictorias: **bloquea la publicación**. Revisa el PNG vs el JSON en `chroma_v2/assets/tables/<doc_id>/` y corrige (re-corre con más presupuesto Vision, o ajusta a mano el JSON), o publica con `--force` (esas tablas se servirán como imagen).

### Paso 4 — Golden set (QA automático, opcional pero recomendado)

```bash
copy pipeline_indexacion\golden\queries.example.json pipeline_indexacion\golden\queries.json
```

Edita `queries.json` con consultas reales y el texto que DEBE aparecer en el top-4 (`expect_contains`). A partir de entonces, cada build calcula `recall@4` contra el índice nuevo y **bloquea la publicación si baja de 0.80** — una regresión del parser se detecta aquí, no en producción.

### Paso 5 — Publicar a MinIO

Hay **dos vías**. Si la conexión a MinIO de Railway es inestable (timeouts/SSL al subir los ~620 MB / 637 archivos), usa la **vía manual** (5B) — es más robusta y reanudable.

#### 5A — Automática (`publish`)

```bash
python -m pipeline_indexacion.v2.run publish
```

1. Sube `chroma_v2/` completo (índice + `assets/` + `assets_registry.sqlite` + manifest + reporte) a un prefijo **versionado**: `chroma_db_versions/vYYYYMMDD-HHMMSS/`.
2. Solo al completar, actualiza el puntero `chroma_db_versions/latest.json`.
3. **Espeja al prefijo legacy** (`data/chroma_db/` por defecto) — es lo que descarga el bot actual (`bootstrap_sync`) al arrancar, sin cambios de código.

```bash
python -m pipeline_indexacion.v2.run publish --force      # publica aunque haya gates fallidos
python -m pipeline_indexacion.v2.run publish --no-mirror  # solo versión, sin tocar lo que ve el bot
python -m pipeline_indexacion.v2.run publish --version v20260612-201021   # REANUDA una subida interrumpida
python -m pipeline_indexacion.v2.run versions             # lista versiones publicadas
python -m pipeline_indexacion.v2.run all                  # build + publish en un solo comando
```

> **Si la subida se corta** (timeouts/SSL contra Railway — frecuente con ~600 MB): re-ejecuta con `--version <la versión del log>`. Los archivos ya subidos (mismo tamaño) se saltan y cada archivo se reintenta hasta 6 veces con backoff y reconexión. El espejo legacy también es reanudable: sube/sobrescribe primero y poda lo obsoleto al final (nunca deja al bot sin índice a mitad de subida).

#### 5B — Manual (`package`) — recomendada si la conexión es inestable

Genera la carpeta lista para subir (no toca la red) y un instructivo con los pasos exactos:

```bash
python -m pipeline_indexacion.v2.run package          # verifica + escribe SUBIR_A_MINIO.txt
python -m pipeline_indexacion.v2.run package --zip     # además crea un .tar.gz de respaldo en dist/
python -m pipeline_indexacion.v2.run package --target-prefix chroma_v2   # prefijo destino (default)
```

`package`:
1. **Verifica** que `chroma_v2/` esté completo (DB + manifests + que TODOS los PNG del registro existan).
2. Reporta el desglose de tamaños: **DB ~107 MB / 10 archivos (crítico)** vs **assets ~513 MB / 628 archivos (pesado)**.
3. Escribe **`chroma_v2/SUBIR_A_MINIO.txt`** con los pasos para subir a `embeddings-store/chroma_v2/` (al lado del v1).

La carpeta a subir es `pipeline_indexacion/chroma_v2/`. Súbela a MinIO al lado del v1 (que está en `data/chroma_db/`). **Sube en 2 pasos** — primero la DB (10 archivos, segundos; ya hace funcionar al bot), luego `assets/` (las imágenes, pueden esperar a la herramienta de tablas):

- **Con `mc` (MinIO Client — recomendado, reanudable):**
  ```bash
  mc alias set railway https://<MINIO_PUBLIC_ENDPOINT> ACCESS_KEY SECRET_KEY
  mc mirror --overwrite "pipeline_indexacion/chroma_v2" railway/embeddings-store/chroma_v2
  ```
  `mc mirror` reintenta y reanuda solo lo que falte — ideal para los 628 archivos.
- **Con la consola web de MinIO:** entra al bucket `embeddings-store`, crea/entra a `chroma_v2/`, arrastra primero los archivos de la DB y luego la carpeta `assets/`.

**Activar v2 en el bot:** en Railway, servicio del bot → `MINIO_PREFIX=chroma_v2` → redeploy. Rollback instantáneo: `MINIO_PREFIX=data/chroma_db`.

> **Mantener el arranque del bot ligero (opcional):** si no quieres que el bot descargue los 513 MB de imágenes en cada arranque, sube `assets/` a un prefijo aparte (p. ej. `embeddings-store/assets_v2/`) en vez de dentro de `chroma_v2/`. El bot solo necesita la DB para responder texto; la futura herramienta de tablas leerá las imágenes desde su prefijo.

**Rollback:** las versiones nunca se borran; basta reapuntar `latest.json` (y re-espejar la versión buena al prefijo legacy).

### Paso 6 — Verifica en el bot

Redeploy del bot (Railway) → al arrancar, `bootstrap_sync` descarga el índice nuevo. Verifica en logs: `colección 'normativas': N items` con el conteo del reporte, y prueba una consulta de tabla.

---

## Cuándo re-indexar

- Se agrega/actualiza un PDF en `docs/` (el hash lo detecta; los demás documentos se saltan).
- Cambia el modelo de embeddings → borra `v2_work/embcache.sqlite` y corre `build`.
- Cambian parámetros de chunking (`V2_CHUNK_TOKENS`) → `build` (los embeddings de chunks idénticos salen del caché).
- Quieres mejorar tablas `solo_imagen`/`revision` → `build` con más `V2_VISION_MAX_CALLS` (solo re-paga las pendientes).

---

## Estructura de salidas

```
pipeline_indexacion/
├── v2_work/                      # intermedios persistentes (NO se publica, gitignored)
│   ├── inventory/{doc_id}.json       # perfil por página + hash (incrementalidad)
│   ├── text/{doc_id}.json            # texto limpio por página
│   ├── structure/{doc_id}.toc.json   # tabla de contenido detectada
│   ├── assets/tables|figures/...     # PNG + JSON canónicos (caché entre corridas)
│   ├── assets_registry.sqlite        # registro de activos (fuente)
│   └── embcache.sqlite               # caché de embeddings
│
└── chroma_v2/                    # ÍNDICE FINAL (se publica completo, gitignored)
    ├── chroma.sqlite3 + <uuid>/      # ChromaDB, colección "normativas", coseno, 1536 dims
    ├── assets/tables/{doc_id}/{tabla_id}.png|.json
    ├── assets/figures/{doc_id}/{figura_id}.png
    ├── assets_registry.sqlite        # copia publicada del registro
    ├── index_manifest.json           # modelo, dims, conteos, docs
    └── index_report.json             # reporte de calidad + gates
```

### En MinIO

```
embeddings-store/
├── data/chroma_db/                   # espejo legacy: lo que descarga el bot hoy
│   └── (contenido completo de chroma_v2/)
└── chroma_db_versions/
    ├── latest.json                   # puntero a la versión activa
    └── v20260612-180000/             # cada publicación, inmutable (rollback)
```

---

## El registro de activos en runtime

`assets_registry.sqlite` viaja dentro del índice, así que tras el sync del bot queda en `$CHROMA_DB_DIR/assets_registry.sqlite`. Consulta exacta de una tabla:

```python
from pipeline_indexacion.v2.registry import AssetRegistry
import os

reg = AssetRegistry(os.path.join(os.environ["CHROMA_DB_DIR"], "assets_registry.sqlite"))
t = reg.find_table_any_doc("220.55")[0]
# t["estado"]    → verificada | extraida | solo_imagen | revision
# t["markdown"]  → tabla entregable en chat (si se extrajo)
# t["png_key"]   → assets/tables/.../220.55.png (recorte exacto, fallback de fidelidad)
# t["pagina_inicio"], t["bbox"] → ubicación exacta en el PDF
# t["telegram_file_id"] → columna reservada para cachear el reenvío en Telegram
```

Los chunks sintéticos de tablas/figuras en Chroma llevan metadata `tipo="tabla"|"figura"`, `tabla_id`/`figura_id` y `png_key`, de modo que el retrieval semántico también puede llevar al activo (la integración del tool `extraer_tabla` en el agente está especificada en `mejoras_propuestas.md` §3).

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Falta MINIO_ENDPOINT` en publish | `.env` sin variables MinIO | Configurar `.env` raíz (o `pipeline_indexacion/.env`) |
| `SSLEOFError` / `ConnectTimeoutError` / `MaxRetryError` durante publish | conexión inestable con el edge de Railway (subida grande) | Re-ejecutar `publish --version <vXXXX del log>` — reanuda saltando lo ya subido |
| `[SKIP] PDF no encontrado` | archivo faltante en `docs/` | Copiarlo a `pipeline_indexacion/docs/` |
| `OPENAI_API_KEY no configurada — Vision no disponible` | falta la key | Configurarla, o correr con `--no-vision` |
| Muchas tablas `solo_imagen` | presupuesto Vision agotado (ver warning en logs) | Subir `V2_VISION_MAX_CALLS` y re-correr `build` (solo paga las pendientes) |
| `GATE: N tablas en estado 'revision'` | geometría y Vision discrepan | Revisar PNG vs JSON en `chroma_v2/assets/tables/`; corregir o `publish --force` |
| `GATE: recall del golden set < 0.80` | regresión de extracción/chunking | Comparar `index_report.json` con la corrida anterior; revisar el doc afectado |
| Texto de un doc quedó viejo tras editar el parser | caché de texto por hash | `build --force-text` |
| Cambié de modelo de embeddings | caché con vectores del modelo anterior | Borrar `v2_work/embcache.sqlite` |
| El bot no ve el índice nuevo | publicado con `--no-mirror` | `publish` normal (espeja a `data/chroma_db/`) y redeploy |

---

## Legacy (v1)

`indexar_normativas.py` se conserva solo como fuente del `DOCUMENT_REGISTRY` y de `resolver_doc_id()`. No lo uses para indexar: produce el texto de la NTC con las columnas entrelazadas y no extrae tablas ni figuras.
