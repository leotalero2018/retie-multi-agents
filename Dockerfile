# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# -------------------------------
# System deps (only what you need)
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

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
# - requirements.txt del repo
# - más: langgraph + langchain + langchain-openai (para CallbackHandler)
# - minio client
# -------------------------------
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt --progress-bar off && \
    pip install --no-cache-dir \
        langchain==0.3.26 \
        langchain-openai==0.2.14 \
        langgraph==1.0.2 \
        minio==7.2.16


# -------------------------------
# Copy source
# -------------------------------
COPY . .

# -------------------------------
# Create runtime dirs (writable)
# -------------------------------
RUN mkdir -p /data/chroma_db /app/downloads && \
    chmod -R 777 /data

