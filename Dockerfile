# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# -------------------------------
# Dependencias del sistema
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Directorio de trabajo
# -------------------------------
WORKDIR /app

# -------------------------------
# Variables de entorno útiles
# -------------------------------
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5 \
    PYTHONPATH=/app \
    CHROMA_DB_DIR=/app/data/chroma_db \
    COLLECTION_NAME=retie_docs \
    EMBEDDING_MODEL=text-embedding-3-small \
    CHAT_MODEL=gpt-4o-mini

# -------------------------------
# Copiar requirements primero para usar cache
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Instalar dependencias
# -------------------------------
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --progress-bar off

# -------------------------------
# Copiar código fuente
# -------------------------------
COPY . .

# -------------------------------
# Crear carpetas necesarias
# -------------------------------
RUN mkdir -p downloads data/chroma_db

# -------------------------------
# Exponer puerto (Railway requiere que el contenedor escuche en $PORT)
# -------------------------------
EXPOSE 8000

# -------------------------------
# CMD principal: API con FastAPI (uvicorn)
# -------------------------------
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
