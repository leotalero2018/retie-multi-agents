# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# -------------------------------
# System deps
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Node.js 20 LTS (for notebooklm-mcp)
# -------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Pre-install notebooklm-mcp globally so npx doesn't download it on every boot
RUN npm install -g notebooklm-mcp@latest

# HOTFIX (issue #50 de notebooklm-mcp, sin resolver en 2.0.0): el selector
# `.to-user-container:last-child` deja de coincidir cuando NotebookLM añade
# cualquier elemento después del último contenedor de respuesta (cambio de UI
# de junio 2026), y ask_question expira aunque la respuesta sea visible.
# Se quita el `:last-child`; el código ya usa `.last()` para tomar la más
# reciente. Eliminar cuando upstream publique la corrección.
RUN sed -i 's/latestAnswerText: "\.to-user-container:last-child \.message-text-content"/latestAnswerText: ".to-user-container .message-text-content"/' \
    "$(npm root -g)/notebooklm-mcp/dist/notebooklm/selectors.js" && \
    grep -q 'latestAnswerText: "\.to-user-container \.message-text-content"' \
    "$(npm root -g)/notebooklm-mcp/dist/notebooklm/selectors.js"

# notebooklm-mcp usa Patchright (fork de Playwright). En headless lanza
# `chrome-headless-shell`, que `playwright install chromium` NO instala y cuyo
# revision (p. ej. chromium_headless_shell-1223) debe coincidir con el de Patchright.
# Sin esto ask_question falla: "Executable doesn't exist at
# .../chromium_headless_shell-XXXX/chrome-headless-shell". Instalamos los
# navegadores con el Patchright QUE TRAE notebooklm-mcp para que el revision
# coincida (fallback a `npx patchright` y, como último recurso, a playwright).
RUN NLM="$(npm root -g)/notebooklm-mcp"; \
    PR="$NLM/node_modules/.bin/patchright"; \
    [ -x "$PR" ] || PR="npx --yes patchright"; \
    $PR install --with-deps chromium; \
    $PR install chromium-headless-shell \
      || npx --yes playwright install chromium-headless-shell \
      || true

# -------------------------------
# Workdir
# -------------------------------
WORKDIR /app

# -------------------------------
# Env (align CHROMA_* paths + enable Langfuse, disable LangSmith)
# *NO* metas secretos en la imagen; pásalos en runtime (Railway/ENV).
# -------------------------------
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHROMA_DB_DIR=/data/chroma_db \
    CHROMA_PERSIST_DIR=/data/chroma_db \
    COLLECTION_NAME=retie_docs \
    EMBEDDING_MODEL=text-embedding-3-small \
    CHAT_MODEL=gpt-4o-mini \
    CHROMADB_TELEMETRY=OFF \
    OCR_LANG=spa+eng \
    VISION_MODEL=gpt-4o \
    LANGFUSE_ENABLED=true \
    # El SDK de Langfuse lee estas 3 vars en runtime:
    LANGFUSE_BASE_URL=https://cloud.langfuse.com

# -------------------------------
# Copy requirements first (better cache)
# -------------------------------
COPY requirements.txt ./requirements.txt

# -------------------------------
# Install deps
# - requirements.txt es la ÚNICA fuente de pines. Antes se reinstalaban aquí
#   langgraph/langchain/minio con versiones VIEJAS (langgraph 0.2.42 pisaba el
#   0.2.62 de requirements) y langchain-openai, que no se importa en ninguna
#   parte (el CallbackHandler de Langfuse viene de langfuse.langchain).
# -------------------------------
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt --progress-bar off



# -------------------------------
# Copy source
# -------------------------------
COPY . .

# -------------------------------
# Create runtime dirs (writable)
# -------------------------------
RUN mkdir -p /data/chroma_db /app/downloads && \
    chmod -R 777 /data

