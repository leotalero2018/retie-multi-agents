# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# Dependencias del sistema (mantener mínimo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Actualiza pip primero (reduce problemas de resolución), y establece variables útiles
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

# 2) Copia requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# 3) Instala dependencias
# Si no agregaste el --extra-index-url en requirements.txt, puedes exportarlo aquí:
# ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --progress-bar off

# 4) Copia el código fuente
COPY . .

# 5) Asegura que existan los directorios de runtime
RUN mkdir -p downloads data/chroma_db

# 6) Variables de entorno de la app
ENV CHROMA_DB_DIR=data/chroma_db \
    COLLECTION_NAME=retie_docs \
    EMBEDDING_MODEL=text-embedding-3-small \
    CHAT_MODEL=gpt-4o-mini \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
