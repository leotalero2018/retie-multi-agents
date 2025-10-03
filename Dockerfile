# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

TEST-DEBUG, SE CONGELA EL SERVICIO POR EL MOMENTO

# -------------------------------
# System deps (include OCR/FFmpeg only if you use them)
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Workdir
# -------------------------------
WORKDIR /app

# -------------------------------
# Useful env (align both CHROMA_* paths)
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
    CHAT_MODEL=gpt-4o-mini

# -------------------------------
# Copy requirements first (better cache)
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Install deps (explicitly ensure minio is present)
# -------------------------------
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt --progress-bar off && \
    pip install --no-cache-dir minio==7.2.16

# -------------------------------
# Copy source
# -------------------------------
COPY . .

# -------------------------------
# Create runtime dirs
# -------------------------------
RUN mkdir -p /data/chroma_db /app/downloads

# -------------------------------
# Start command is provided by Railway (e.g. python -m app.bot.run_polling)
# -------------------------------
