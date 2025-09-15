FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p downloads data/chroma_db

ENV CHROMA_DB_DIR=data/chroma_db \
    COLLECTION_NAME=retie_docs \
    EMBEDDING_MODEL=text-embedding-3-small \
    CHAT_MODEL=gpt-4o-mini \
    PYTHONPATH=/app

# Exponemos 8000 para local; en Railway se usa $PORT
EXPOSE 8000

# Usa uvicorn y respeta $PORT (fallback 8000 en local)
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
