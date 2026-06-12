# Pipeline de Indexación de Normativas Eléctricas

Indexa los PDFs de normativas eléctricas colombianas en ChromaDB y sincroniza la base de datos con MinIO (Railway).

---

## Documentos indexados

| doc_id | Archivo | Colección |
|---|---|---|
| `ntc2050_v2` | NTC 2050 V2 - Código Eléctrico Colombiano | `norma_vigente` |
| `ntc2050_erratas` | NTC 2050 - Fe de Erratas | `norma_vigente` |
| `retie_libro1` | RETIE Libro 1 - Disposiciones Generales | `normas_historicas` |
| `retie_libro2` | RETIE Libro 2 - Productos | `normas_historicas` |
| `retie_libro3` | RETIE Libro 3 - Instalaciones | `normas_historicas` |
| `retie_libro4` | RETIE Libro 4 - Evaluación de la Conformidad | `normas_historicas` |

Los PDFs deben estar en `pipeline_indexacion/docs/`.

---

## Requisitos previos

### 1. Python 3.10+

Verifica con:
```bash
python --version
```

### 2. Dependencias

Desde la raíz del proyecto:
```bash
pip install chromadb langchain langchain-community pypdf sentence-transformers minio python-dotenv
```

> La primera ejecución descarga el modelo de embeddings `paraphrase-multilingual-mpnet-base-v2` (~440 MB). Las siguientes lo usan desde caché local.

### 3. Credenciales MinIO

El script lee credenciales del `.env` en la raíz del proyecto. Asegúrate de tener estas variables configuradas:

```env
MINIO_PUBLIC_ENDPOINT=<host:puerto>     # ej: bucket.railway.app:443
MINIO_ROOT_USER=<access_key>
MINIO_ROOT_PASSWORD=<secret_key>
```

Encuéntralas en Railway → servicio MinIO → pestaña **Variables**.

Si prefieres configurarlas solo para este script, crea `pipeline_indexacion/.env`:
```env
MINIO_ENDPOINT=<host:puerto>
MINIO_ACCESS_KEY=<access_key>
MINIO_SECRET_KEY=<secret_key>
MINIO_SECURE=true
```

---

## Ejecución

Desde la **raíz del proyecto**:

```bash
python pipeline_indexacion/indexar_normativas.py
```

### Flujo que ejecuta el script

```
1. Carga credenciales desde .env
2. Conecta a MinIO y verifica que el bucket "embeddings-store" exista
3. Elimina data/chroma_db/ del bucket (limpia la indexación anterior)
4. Descarga/carga el modelo de embeddings (HuggingFace, local)
5. Por cada PDF:
   a. Carga el archivo con PyPDFLoader
   b. Divide en chunks (800 tokens, overlap 150)
   c. Agrega metadata: doc_id, colección, página, alias, vigencia...
   d. Indexa en ChromaDB local (pipeline_indexacion/chroma_local/)
6. Imprime resumen de chunks por colección
7. Sube todos los archivos de chroma_local/ a MinIO → data/chroma_db/
8. Confirma total de archivos subidos
```

### Salida esperada

```
10:32:01 [INFO] Conectando a MinIO en bucket.railway.app:443 ...
10:32:02 [INFO] Eliminando embeddings-store/data/chroma_db/ ...
10:32:02 [INFO]   47 objetos eliminados.
10:32:02 [INFO] Cargando modelo de embeddings: sentence-transformers/...
10:32:15 [INFO] Procesando: NTC 2050 V2 - Código Eléctrico Colombiano
10:32:15 [INFO]   3842 chunks de 612 páginas
...
10:48:30 [INFO] ── Resumen de indexación ──
10:48:30 [INFO]   norma_vigente          4105 chunks
10:48:30 [INFO]   normas_historicas      8731 chunks
10:48:35 [INFO]   ✅ 53 archivos subidos a MinIO.
10:48:35 [INFO] Indexación completa.
```

**Tiempo estimado:** 15–30 minutos dependiendo del hardware.

---

## Cuándo re-indexar

Ejecuta el script de nuevo cuando:

- Se agregue o actualice algún PDF en `docs/`
- Cambie el modelo de embeddings (`EMBEDDING_MODEL` en el script)
- Cambien los parámetros de chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
- Se modifique la metadata de algún documento

El script **siempre limpia la indexación anterior** antes de generar la nueva.

---

## Estructura de ChromaDB en MinIO

```
embeddings-store/
└── data/
    └── chroma_db/
        ├── chroma.sqlite3
        ├── <uuid-norma_vigente>/
        │   ├── data_level0.bin
        │   └── ...
        └── <uuid-normas_historicas>/
            ├── data_level0.bin
            └── ...
```

---

## Función auxiliar: resolver doc_id

El script expone `resolver_doc_id()` para uso programático:

```python
from pipeline_indexacion.indexar_normativas import resolver_doc_id

resolver_doc_id("dame el libro 3")   # → "retie_libro3"
resolver_doc_id("la norma actual")   # → "ntc2050_v2"
resolver_doc_id("fe de erratas")     # → "ntc2050_erratas"
resolver_doc_id("texto sin match")   # → None
```

---

## Solución de problemas

| Error | Causa probable | Solución |
|---|---|---|
| `Falta MINIO_ENDPOINT` | Variables de entorno no cargadas | Verificar `.env` en raíz o en `pipeline_indexacion/` |
| `[SKIP] No encontrado: archivo.pdf` | PDF faltante en `docs/` | Copiar el archivo a `pipeline_indexacion/docs/` |
| `ModuleNotFoundError: chromadb` | Dependencia no instalada | `pip install chromadb langchain-community` |
| `OSError: [Errno 110]` (MinIO timeout) | Endpoint incorrecto o sin conectividad | Verificar `MINIO_PUBLIC_ENDPOINT` y conexión a internet |
| Error de CUDA / torch | GPU no disponible | Normal en CPU; el script funciona igual (más lento) |
