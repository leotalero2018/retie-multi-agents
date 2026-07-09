<div align="center">

# RETIE Multi-Agents

**Sistema RAG multiagente para consultas técnicas sobre el Reglamento Técnico de Instalaciones Eléctricas (RETIE)**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.62-FF6B35?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.1.0-FF4081?style=for-the-badge)](https://www.trychroma.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---
![Diagrama general](images/diagrama.png)

---

## Arquitectura general

```
Usuario
  │
  ├── Telegram Bot (voz, texto, imagen)
  │       └── retie_agent/bot/
  │
  ├── API REST (HTTP)
  │       └── retie_agent/api/
  │
  └── CLI (script local)
          └── index_docs.py
                │
          ┌─────▼────────────────────────────────────────┐
          │               LangGraph Pipeline              │
          │   retrieve → router → answer → enrich → style │
          │          retie_agent/agent/graph.py           │
          └─────────────────┬────────────────────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
    retie_agent/retriever/        retie_agent/services/
     Chroma (dense + BM25)        MongoDB · MinIO · Whisper
                                  Vision · Langfuse
             ▲
             │  (índice vectorial pre-construido)
             │
    pipeline_indexacion/
     PDF → parsers → chunker → embedder → ChromaDB
```

---

## Estructura del monorepo

```
retie-multi-agents/
│
├── retie_agent/                    ← Agente inteligente RETIE
│   ├── config.py                   # Configuración centralizada (Pydantic Settings)
│   ├── agent/
│   │   ├── graph.py                # Grafo LangGraph (pipeline principal de respuesta)
│   │   ├── retie_agent.py          # Agente base con dedupe y resolución de colecciones
│   │   ├── enrichment_assistant.py # Enriquecimiento con OpenAI Assistants API
│   │   ├── registry.py             # Registro de agentes disponibles (plumber, pymupdf)
│   │   ├── orchestrator.py         # Wrapper de compatibilidad
│   │   └── prompt.py               # Templates de prompts (usuario vs admin)
│   ├── api/
│   │   ├── main.py                 # FastAPI: /health, /query, /files, /gallery
│   │   ├── ingest.py               # Endpoints /ingest/rebuild y /ingest/sync
│   │   └── debug.py                # Endpoint /debug/bucket (S3)
│   ├── bot/
│   │   ├── run_polling.py          # Entry point del bot Telegram (aiogram)
│   │   └── router.py               # Handlers: texto, voz, imágenes, comandos admin
│   ├── llm/
│   │   ├── provider.py             # OpenAI client + respuesta extractiva fallback
│   │   └── embedder.py             # Generación de embeddings (text-embedding-3-small)
│   ├── retriever/
│   │   ├── retrieve.py             # Dense retrieval + BM25 fallback
│   │   └── chroma_client.py        # Cliente persistente ChromaDB (singleton)
│   ├── services/
│   │   ├── history.py              # Historial de chat en MongoDB
│   │   ├── mongo_store.py          # Almacenamiento de imágenes (GridFS)
│   │   ├── storage_minio.py        # Cliente MinIO/S3
│   │   ├── vision.py               # OCR (Tesseract) + GPT-4o Vision
│   │   └── whisper.py              # Transcripción de audio (Whisper / GPT-4o)
│   ├── observability/
│   │   └── obs.py                  # Trazabilidad con Langfuse v3
│   ├── db/
│   │   └── chroma.py               # Adaptador minimalista de ChromaDB
│   └── utils/
│       ├── text.py                 # Tokenización, limpieza y splitting de texto
│       └── storage.py              # Helpers S3-compatibles (boto3)
│
├── pipeline_indexacion/            ← Pipeline de indexación de documentos
│   ├── ingestion/
│   │   ├── pipeline.py             # index_folder(): indexa una carpeta completa de PDFs
│   │   ├── indexer.py              # index_file(): indexa un PDF individual
│   │   ├── chunker.py              # Splitting por tokens con overlap (tiktoken)
│   │   └── parsers/
│   │       ├── base.py             # BaseParser (interfaz)
│   │       ├── factory.py          # Selección dinámica de parser
│   │       ├── pdfplumber_parser.py# Parser preciso para tablas y layouts complejos
│   │       └── pymupdf_parser.py   # Parser rápido para texto plano
│   ├── bootstrap_sync.py           # Descarga automática del índice Chroma desde MinIO
│   └── bootstrap_preflight.py      # Self-test del RAG al arrancar
│
├── tests/                          ← Suite de pruebas
│   ├── test_chuncker.py
│   ├── test_history.py
│   ├── check_indexed_pdfs.py
│   ├── test_pipeline_minio.py
│   ├── test_minio.py
│   └── test_search_minio.py
│
├── docs/                           ← Documentos PDF normativos (no versionados)
├── data/                           ← Base de datos ChromaDB persistente (no versionado)
├── images/                         ← Imágenes del README
│
├── index_docs.py                   # CLI para indexar PDFs desde terminal
├── check_counts.py                 # Verifica el número de chunks indexados
├── test_langfuse.py                # Test de conexión con Langfuse
│
├── Dockerfile                      # Imagen Docker multi-etapa
├── docker-compose.yml              # Servicios: bot + MinIO
├── Procfile                        # Para Railway: `python -m retie_agent.bot.run_polling`
├── requirements.txt                # Dependencias Python
└── .env                            # Variables de entorno (no se versiona)
```

---

## Stack tecnológico

| Categoría | Tecnología | Versión |
|-----------|-----------|---------|
| **Framework web** | FastAPI + Uvicorn | 0.112 / 0.30 |
| **Bot Telegram** | aiogram | 3.7.0 |
| **Orquestación de agentes** | LangGraph + LangChain | 0.2.62 / 0.3.25 |
| **Base de datos vectorial** | ChromaDB (HNSW cosine) | 1.1.0 |
| **LLM** | OpenAI GPT-4o-mini | — |
| **Embeddings** | text-embedding-3-small (1536 dim) | — |
| **Parsers PDF** | pdfplumber + PyMuPDF | 0.11.7 / 1.24.9 |
| **Tokenización** | tiktoken (cl100k_base) | 0.7.0 |
| **Recuperación léxica** | rank-bm25 (fallback) | 0.2.2 |
| **Transcripción de voz** | Whisper / GPT-4o-transcribe | — |
| **Vision / OCR** | GPT-4o + Tesseract | — |
| **Historial de chat** | MongoDB + PyMongo | 4.15.0 |
| **Almacenamiento vectores** | MinIO (S3-compatible) | 7.2.15 |
| **Observabilidad** | Langfuse v3 | 4.7.1+ |
| **Contenedores** | Docker + docker-compose | — |

---

## Prerrequisitos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.11+** — [descargar](https://www.python.org/downloads/)
- **Git** — [descargar](https://git-scm.com/)
- **Node.js 20+** (para el servidor NotebookLM MCP)
  - macOS: `brew install node`
  - Ubuntu: `sudo apt install nodejs npm` (o [nvm](https://github.com/nvm-sh/nvm))
  - Windows: [instalador](https://nodejs.org/)
- **Docker + Docker Compose** (opcional, para ejecutar con contenedores)
- **Tesseract OCR** (opcional, para procesar imágenes)
  - Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-spa`
  - Windows: [instalador](https://github.com/UB-Mannheim/tesseract/wiki)
- **ffmpeg** (opcional, para normalizar audio)
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: [descargar](https://ffmpeg.org/download.html)

Cuentas necesarias:
- **OpenAI** con API Key activa ([platform.openai.com](https://platform.openai.com))
- **Telegram Bot Token** (créalo con [@BotFather](https://t.me/BotFather))
- **MongoDB** — [MongoDB Atlas](https://www.mongodb.com/atlas) (gratuito) o instancia local
- **Langfuse** (opcional, para observabilidad) — [cloud.langfuse.com](https://cloud.langfuse.com)

---

## Inicio rápido

### 1. Clonar el repositorio

```bash
git clone https://github.com/leotalero2018/retie-multi-agents.git
cd retie-multi-agents
```

### 2. Crear y activar el entorno virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copia el archivo de ejemplo y edítalo con tus credenciales:

```bash
cp .env.example .env   # Linux/macOS
copy .env.example .env  # Windows
```

Abre `.env` y configura las variables (ver sección [Variables de entorno](#variables-de-entorno)).

### 5. Configurar el PYTHONPATH

El proyecto usa dos paquetes raíz (`retie_agent` y `pipeline_indexacion`), por lo que debes apuntar el PYTHONPATH a la raíz del repositorio.

**Linux / macOS:**
```bash
export PYTHONPATH=$(pwd)
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "$(Get-Location)"
```

> **Tip:** Agrega esta línea a tu `.bashrc`, `.zshrc` o al perfil de PowerShell para no repetirla cada sesión.

---

## Variables de entorno

Crea un archivo `.env` en la raíz con las siguientes variables:

```env
# ── OpenAI ─────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...          # API key principal
CHROMA_OPENAI_API_KEY=sk-proj-...   # (opcional) key exclusiva para embeddings

# ── LLM / Embeddings ───────────────────────────────────────────────────
CHAT_MODEL=gpt-4o-mini              # Modelo de chat
EMBEDDING_MODEL=text-embedding-3-small
MAX_TOKENS=600

# ── Telegram ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=1234567890:AAE...

# ── RAG / ChromaDB ─────────────────────────────────────────────────────
COLLECTION_NAMES=normativas         # colección única que consulta el agente
CHROMA_PERSIST_DIR=./data/chroma_db
CHUNK_TOKENS=800                    # Tokens por chunk
CHUNK_OVERLAP=150                   # Overlap entre chunks
TOP_K=8                             # Documentos recuperados por query
RAG_DISTANCE_THRESHOLD=0.45         # Umbral de similitud coseno
BM25_FUSION_ENABLED=true            # Fusión léxica BM25 + dense vía RRF
FETCH_K_MULTIPLIER=3                # Candidatos dense extra antes de fusionar
QUERY_REWRITE_ENABLED=true          # Reescribe preguntas de seguimiento usando el historial
HISTORY_LIMIT=10                    # Mensajes de historial por sesión

# ── NotebookLM (híbrido) ───────────────────────────────────────────────
NOTEBOOKLM_ENABLED=true
NOTEBOOKLM_ALWAYS_WAIT=true         # true = SIEMPRE Chroma + NLM, espera completa
NOTEBOOKLM_HARD_TIMEOUT=180         # Tope absoluto de espera a NLM (always_wait)
NLM_SEMANTIC_CACHE_SIM=0.93         # Reusa respuesta NLM de pregunta similar (0=off)
NOTEBOOKLM_MIN_QUERY_CHARS=12       # Preguntas más cortas no consultan NLM

# Espera adaptativa — SOLO aplica con NOTEBOOKLM_ALWAYS_WAIT=false:
NOTEBOOKLM_PARALLEL_TIMEOUT=25      # Espera máx. a NLM sin evidencia Chroma
NOTEBOOKLM_SOFT_TIMEOUT=12          # Espera si Chroma ya tiene evidencia decente
NOTEBOOKLM_TABLE_TIMEOUT=45         # Espera para tablas (dependen de NLM)
CHROMA_HIGH_CONFIDENCE_THR=0.25     # Mejor distancia < esto → no se espera a NLM
CHROMA_DECENT_THR=0.40              # Mejor distancia < esto → espera corta (soft)

# ── Enriquecimiento ────────────────────────────────────────────────────
ENRICHMENT_ENABLED=true             # false = omite el assistant (~2x más rápido)
ENRICHMENT_TIMEOUT=30               # Segundos máx. de espera del assistant

# ── MongoDB ────────────────────────────────────────────────────────────
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=retie
MONGO_BUCKET=images

# ── MinIO / S3 ─────────────────────────────────────────────────────────
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET_NAME=embeddings-store
MINIO_PRIVATE_ENDPOINT=localhost:9000
MINIO_PUBLIC_ENDPOINT=localhost:9000

# ── Langfuse (observabilidad, opcional) ────────────────────────────────
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com

# ── Admin Bot ──────────────────────────────────────────────────────────
ADMIN_PASSWORD=tu_contraseña_segura
ADMIN_USER_IDS=123456789            # IDs Telegram separados por coma

# ── Asistente de enriquecimiento ───────────────────────────────────────
ENRICHMENT_ASSISTANT_ID=asst_...
ENRICHMENT_VECTOR_STORE_ID=vs_...

# ── Vision / OCR ───────────────────────────────────────────────────────
VISION_MODEL=gpt-4o
OCR_LANG=spa+eng
```

---

## Ejecución de cada componente

### Indexar documentos (pipeline_indexacion)

Coloca los PDFs normativos en la carpeta `docs/` y ejecuta:

```bash
# Indexar toda la carpeta (parser por defecto: PyMuPDF)
python index_docs.py --source ./docs --out ./data/chroma_db --engine pymupdf

# Indexar con pdfplumber (mejor para tablas)
python index_docs.py --source ./docs --out ./data/chroma_db --engine pdfplumber

# Indexar un único PDF
python index_docs.py --file ./docs/Resolución_40117_de_2024_RETIE.pdf --engine pymupdf

# Indexar y subir la DB a MinIO/S3
python index_docs.py --source ./docs --out ./data/chroma_db --upload --s3-prefix chroma_db/
```

Parámetros disponibles:

| Parámetro | Descripción | Por defecto |
|-----------|-------------|-------------|
| `--source` | Carpeta con PDFs a indexar | `./docs` |
| `--out` | Directorio de persistencia de ChromaDB | `./data/chroma_db` |
| `--engine` | Parser PDF: `pymupdf` o `pdfplumber` | `pymupdf` |
| `--file` | Indexar un PDF específico | — |
| `--upload` | Subir la DB a S3/MinIO tras indexar | `false` |
| `--s3-prefix` | Prefijo en el bucket S3 | `chroma_db/` |

---

### Bot de Telegram (retie_agent)

```bash
# Activar entorno virtual y PYTHONPATH primero
python -m retie_agent.bot.run_polling
```

Al arrancar, el bot:
1. Sincroniza automáticamente la base ChromaDB desde MinIO (si está configurado)
2. Verifica las colecciones disponibles
3. Inicia el polling de Telegram

**Comandos disponibles en el bot:**

| Comando | Descripción |
|---------|-------------|
| `/start` | Inicia la conversación |
| `/help` | Lista de comandos y capacidades del bot |
| `/clear` | Borra el historial de la conversación actual |
| `/admin <contraseña>` | Acceso administrador (activa citas y fuentes) |
| `/docs [k]` | Muestra los k documentos más relevantes (solo admin) |
| `/logout` | Cierra la sesión de administrador |

Además acepta: mensajes de texto, notas de voz, fotos e imágenes de documentos.

---

### API REST (retie_agent)

```bash
# Iniciar el servidor FastAPI
uvicorn retie_agent.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Accede a la documentación interactiva en: **http://localhost:8000/docs**

**Endpoints principales:**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado del servicio |
| `GET` | `/query?q=...&agent_key=plumber&session_id=...` | Consulta al agente |
| `POST` | `/ingest/rebuild?secret=...&engine=pymupdf` | Reindexar PDFs locales |
| `POST` | `/ingest/sync?secret=...&prefix=chroma_db/` | Sincronizar DB desde S3 |
| `GET` | `/files` | Listar imágenes guardadas (GridFS) |
| `GET` | `/files/{id}` | Descargar imagen por ID |
| `GET` | `/gallery` | Galería HTML de imágenes |
| `GET` | `/debug/bucket?prefix=chroma_db/` | Listar objetos en S3 |
| `POST` | `/telegram/webhook` | Webhook de Telegram (producción) |

**Ejemplo de consulta:**

```bash
curl "http://localhost:8000/query?q=¿Qué%20exige%20el%20RETIE%20sobre%20puesta%20a%20tierra?&agent_key=plumber"
```

```json
{
  "question": "¿Qué exige el RETIE sobre puesta a tierra?",
  "answer": "El RETIE establece que...",
  "session_id": "api_query",
  "agent_key": "plumber"
}
```

### Servidor NotebookLM MCP (retrieval híbrido)

El agente combina ChromaDB con **NotebookLM** a través del servidor MCP [`notebooklm-mcp`](https://www.npmjs.com/package/notebooklm-mcp) (Node.js), que controla NotebookLM con un navegador Playwright autenticado con tu cuenta de Google. Requiere **Node.js 20+**.

> ⚠️ **La sesión de Google caduca periódicamente** (en cuestión de horas si el servidor no está corriendo con el keepalive activo). Si el agente empieza a responder solo con Chroma o ves en los logs `Could not find NotebookLM chat input`, la sesión expiró: repite el paso 3 (re-autenticación) y vuelve a subirla a MinIO (paso 5).

#### 1. Instalar el servidor

```bash
npm install -g notebooklm-mcp@latest
```

#### 2. Arrancar el servidor (déjalo corriendo en su propia terminal)

**Windows (PowerShell):**
```powershell
notebooklm-mcp --transport http --port 3000
```

**macOS / Linux:**
```bash
notebooklm-mcp --transport http --port 3000
```

El servidor guarda la sesión del navegador (cookies de Google + `library.json` con los notebooks registrados) en una carpeta que depende del sistema operativo:

| SO | Carpeta de datos |
|----|------------------|
| Windows | `%LOCALAPPDATA%\notebooklm-mcp\Data` |
| macOS | `~/Library/Application Support/notebooklm-mcp` |
| Linux | `~/.local/share/notebooklm-mcp` |

#### 3. Autenticarse con Google (primera vez o sesión caducada)

Con el servidor corriendo, en **otra terminal** (con el venv activo y `PYTHONPATH` configurado):

```bash
python setup_notebooklm.py
```

El script detecta si la sesión es válida. Si no lo es, **abre una ventana de Chrome** para que inicies sesión con la cuenta de Google que tiene el notebook de RETIE. La ventana se cierra sola al llegar a NotebookLM (tienes hasta 10 minutos). Vuelve a ejecutar el script para confirmar `authenticated = True` y registrar el notebook si hace falta (pega el share link cuando lo pida; el ID queda en `NOTEBOOKLM_NOTEBOOK_ID` del `.env`).

#### 4. (Alternativa) Restaurar una sesión válida desde MinIO

Si otra máquina (o Railway) ya subió una sesión vigente a MinIO, puedes restaurarla sin re-autenticarte. **Detén primero el servidor MCP** y ejecuta:

**Windows (PowerShell):**
```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); os.environ.pop('MINIO_PRIVATE_ENDPOINT', None); from retie_agent.services.nlm_session import download_nlm_session; download_nlm_session()"
```

**macOS / Linux:**
```bash
python -c "
import os
from dotenv import load_dotenv; load_dotenv()
os.environ.pop('MINIO_PRIVATE_ENDPOINT', None)   # el endpoint privado solo existe dentro de Railway
from retie_agent.services.nlm_session import download_nlm_session
download_nlm_session()
"
```

Luego arranca el servidor (paso 2). Si al preguntar sigue fallando, la sesión guardada también caducó → paso 3.

#### 5. Subir la sesión renovada a MinIO (para Railway y otras máquinas)

Cada vez que te re-autentiques, sube la sesión nueva para que Railway la restaure al arrancar:

```bash
python setup_notebooklm.py --upload
# o, si MinIO no es alcanzable, comprime y súbela a mano:
python setup_notebooklm.py --pack
```

#### 6. Verificar la conexión

```bash
python tests/_nlm_check.py
```

Debe mostrar `authenticated = True` y el notebook `retie` en la lista. Asegúrate de tener en el `.env`:

```env
NOTEBOOKLM_ENABLED=true
NOTEBOOKLM_URL=http://localhost:3000
NOTEBOOKLM_NOTEBOOK_ID=retie
```

> 💡 **Keepalive:** con el bot/API corriendo, `NOTEBOOKLM_KEEPALIVE_MINUTES` (default 240) refresca las cookies periódicamente y resube el estado a MinIO, manteniendo viva una sesión válida sin logins manuales. La caducidad ocurre sobre todo cuando el servidor pasa horas apagado.

---

## Spike: Gemini File Search como fuente RAG secundaria (pruebas A/B)

> ⚠️ **Spike experimental — no producción todavía.** Rama `test/gemini-file-search-spike`. Valida si [Gemini File Search](https://ai.google.dev/gemini-api/docs/file-search) (RAG gestionado por API oficial, **sin sesiones ni logins que caducan**) puede reemplazar a NotebookLM como segunda fuente del agente. Con la configuración por defecto el spike no cambia nada del pipeline actual.

### ¿Qué añade?

El grafo ya recuperaba en paralelo con `chromadb_node` + `notebooklm_node`. El spike agrega un **tercer nodo `gemini_node`** que respeta el mismo contrato (escribe en `gemini_docs`, genera su propio span en Langfuse), controlado por un único feature flag:

| `SECONDARY_RAG_SOURCE` | Comportamiento |
|------------------------|----------------|
| `notebooklm` (default) | Pipeline actual, sin cambios. |
| `gemini`               | Gemini File Search alimenta la respuesta; NotebookLM apagado. |
| `shadow`               | **A/B:** ambos corren en paralelo para la misma pregunta; NotebookLM alimenta la respuesta (producción segura) y Gemini queda registrado en su span de Langfuse para comparar calidad. |

El flag se alterna por entorno en Railway sin redeploy de código.

### Requisitos

```bash
pip install google-genai   # SDK oficial (no está en requirements.txt aún)
```

> ⚠️ Instalar `google-genai` sube `pydantic` por encima del pin de `aiogram` (`<2.8`); en las pruebas ambos siguen funcionando, pero es un punto a resolver antes de producción.

Variables en el `.env` (obtén la API key en [Google AI Studio](https://aistudio.google.com/apikey) — al crearla, elige **"crear en proyecto nuevo"**, no necesitas un proyecto previo):

```env
GEMINI_API_KEY=AQ...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FILE_SEARCH_STORE=            # lo llena el script de ingesta (paso 1)
SECONDARY_RAG_SOURCE=notebooklm      # cámbialo a gemini | shadow para probar
```

### Paso 1 — Cargar el corpus en un File Search Store (una sola vez)

`gemini_client.py` solo **consulta** un store existente; `gemini_ingesta.py` lo crea y lo llena. Ignora todo lo que no sea `*.pdf` y sanea nombres con tildes.

```bash
# crear un store nuevo y subir todos los PDF de la carpeta
python gemini_ingesta.py /ruta/a/carpeta/con/pdfs

# AÑADIR documentos a un store que ya existe (sin crear otro)
python gemini_ingesta.py /ruta/a/carpeta --store fileSearchStores/xxxxx
```

Al terminar imprime la línea `GEMINI_FILE_SEARCH_STORE="fileSearchStores/..."` — cópiala al `.env`.

**El store queda atado a la cuenta de la API key.** Para el piloto con la cuenta
del proyecto (retie), crea el store con ESA key (no la personal), anteponiéndola:

```bash
# macOS / Linux
GEMINI_API_KEY="AQ...key_de_la_cuenta" python gemini_ingesta.py "/ruta/a/RETIE DOCUMENTS"
```

```powershell
# Windows (PowerShell)
cd $HOME\Documents\RETIE\retie-multi-agents
.\.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY="AQ...key_de_la_cuenta"
python gemini_ingesta.py "C:\ruta\a\RETIE DOCUMENTS"
```

### Gestionar los stores (listar, ver documentos, borrar)

`gemini_store_admin.py` administra los stores. Usa la key del `.env`, o antepón
`GEMINI_API_KEY="..."` para operar sobre otra cuenta.

```bash
# listar todos los stores de la cuenta
python gemini_store_admin.py list

# ver los documentos de un store
python gemini_store_admin.py docs fileSearchStores/xxxxx

# borrar un store COMPLETO (con todos sus documentos)
python gemini_store_admin.py delete fileSearchStores/xxxxx

# borrar un solo documento
python gemini_store_admin.py deldoc fileSearchStores/xxxxx fileSearchStores/xxxxx/documents/yyyyy
```

> Para empezar de cero (p. ej. si se subieron archivos equivocados): borra el store
> con `delete` y vuelve a correr `gemini_ingesta.py`. En Windows, antepón la key con
> `$env:GEMINI_API_KEY="..."` en una línea aparte, igual que en la ingesta.

### Paso 2 — Probar en tres niveles

```bash
# Nivel 1 — cliente Gemini aislado (feedback más rápido)
python -c "from dotenv import load_dotenv; load_dotenv(); \
from retie_agent.agent.gemini_client import GeminiFileSearchClient; from retie_agent.config import settings; \
c=GeminiFileSearchClient(api_key=settings.GEMINI_API_KEY, model=settings.GEMINI_MODEL, store=settings.GEMINI_FILE_SEARCH_STORE); \
print(c.ask_question('¿Qué exige el RETIE sobre puesta a tierra?')[0][:400])"

# Nivel 2 — por el grafo, modo gemini
SECONDARY_RAG_SOURCE=gemini python -c "from dotenv import load_dotenv; load_dotenv(); \
from retie_agent.config import settings; settings.SECONDARY_RAG_SOURCE='gemini'; \
from retie_agent.agent.graph import run_graph; \
print(run_graph('¿Qué distancias de seguridad exige el RETIE?', session='gem'))"

# Nivel 3 — shadow A/B: compara notebooklm_node vs gemini_node en Langfuse
# (.env: SECONDARY_RAG_SOURCE=shadow y LANGFUSE_ENABLED=true, luego corre el golden set del TICKET-005)
```

En **Langfuse**, cada traza `retie-query` en modo `shadow` muestra los spans `notebooklm_node` y `gemini_node` lado a lado: se comparan cobertura de filas, exactitud de valores y calidad de citas a igualdad de pregunta.

### Si algo falla

| Síntoma | Solución |
|---------|----------|
| `model not found` | Ajusta `GEMINI_MODEL` en el `.env` (el nombre del modelo pudo cambiar). |
| Falla la ingesta por el modelo de embedding | Cambia `gemini-embedding-2` en `gemini_ingesta.py`. |
| `gemini_node` no aparece en la traza | Falta `GEMINI_API_KEY`/`GEMINI_FILE_SEARCH_STORE`, o `SECONDARY_RAG_SOURCE` sigue en `notebooklm`. |

---

## Despliegue con Docker

### Desarrollo local (bot + MinIO)

```bash
docker-compose up --build
```

Esto levanta:
- **bot**: Contenedor con el bot de Telegram (`retie_agent.bot.run_polling`)
- **minio**: Servidor MinIO en `localhost:9000` (consola en `localhost:9001`)

Accede a la consola de MinIO: **http://localhost:9001** (usuario: `minioadmin` / contraseña: `minioadmin`)

### Solo el bot

```bash
docker build -t retie-agent .
docker run --env-file .env -v $(pwd)/data:/app/data retie-agent python -m retie_agent.bot.run_polling
```

### Solo la API

```bash
docker run --env-file .env -p 8000:8000 -v $(pwd)/data:/app/data retie-agent \
  uvicorn retie_agent.api.main:app --host 0.0.0.0 --port 8000
```

---

## Despliegue en Railway

Este proyecto está preparado para Railway con el `Procfile` incluido:

```
worker: python -m retie_agent.bot.run_polling
```

Variables de entorno a configurar en Railway:
- Todas las de la sección [Variables de entorno](#variables-de-entorno)
- `PYTHONPATH=/app`
- `CHROMA_PERSIST_DIR=/data/chroma_db`

Para el servicio de API, el comando de inicio es:
```
uvicorn retie_agent.api.main:app --host 0.0.0.0 --port $PORT
```

---

## Flujo de datos

### Indexación (pipeline_indexacion)
![Pipeline de indexación](images/pipeline_parser.png)

### Consulta (retie_agent)
![Flujo de consulta](images/flujo_consulta.png)

---

## Tests

```bash
# Verificar chunks indexados
python tests/check_indexed_pdfs.py

# Probar chunking de texto
python -m pytest tests/test_chuncker.py -v

# Probar historial de chat
python -m pytest tests/test_history.py -v

# Verificar conteos en Chroma
python check_counts.py

# Test de Langfuse
python test_langfuse.py
```

---

## Corpus y colección única

Todo el corpus vive en la colección ChromaDB **`normativas`** (NTC 2050 V2 + fe de erratas + RETIE Libros 1-4). No hay cambio de agentes: la respuesta siempre es híbrida (Chroma dense + BM25 + NotebookLM en paralelo) y la vigencia de cada documento queda registrada en la metadata `vigente` de cada chunk.

Para (re)generar el índice y subirlo a MinIO (`embeddings-store/data/chroma_db/`):

```bash
python pipeline_indexacion/indexar_normativas.py
```

> Los embeddings se generan con el mismo modelo que usa el agente (`text-embedding-3-small`, 1536 dims). Si cambias `EMBEDDING_MODEL`, debes re-indexar.

---

## Observabilidad con Langfuse

El proyecto integra Langfuse v3 para trazabilidad completa del pipeline. Actívalo configurando:

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

Cada consulta genera una traza con los nodos: `route_entry → condense_node → retrieve/hybrid_retrieve → answer_node → enrich_node → stylist_node` (más `table_node` cuando se pide una tabla, o `smalltalk_node` para saludos). El span `hybrid_retrieve` registra `nlm_status` (ok / cache_hit_exact / cache_hit_semantic / skipped_high_conf / timeout_soft / timeout / error) y el presupuesto de espera usado.

---

## Contribuir

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feature/mi-mejora`
3. Realiza tus cambios y haz commit: `git commit -m "feat: descripción del cambio"`
4. Abre un Pull Request hacia la rama `dev`

---

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<div align="center">

Desarrollado para facilitar el acceso al conocimiento técnico del RETIE colombiano.

**[Reportar un bug](https://github.com/leotalero2018/retie-multi-agents/issues)** · **[Solicitar una función](https://github.com/leotalero2018/retie-multi-agents/issues)**

</div>
