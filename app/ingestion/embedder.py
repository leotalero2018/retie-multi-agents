# app/ingestion/embedder.py
from typing import List
from app.config import settings

if getattr(settings, "EMBEDDING_PROVIDER", "openai") == "openai":
    # === OpenAI con batching por tokens ===
    from openai import OpenAI
    import tiktoken

    _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    _enc = tiktoken.get_encoding("cl100k_base")

    # Límite seguro por request (OpenAI ~300k). Dejamos margen.
    _MAX_TOKENS_PER_REQ = 280_000

    def _count_tokens(txt: str) -> int:
        return len(_enc.encode(txt))

    def _make_batches(items: List[str]) -> List[List[str]]:
        batches: List[List[str]] = []
        cur_batch: List[str] = []
        cur_tokens = 0

        for t in items:
            tt = _count_tokens(t)
            # Si un solo chunk es gigantesco (raro), igual lo mandamos solo
            if tt > _MAX_TOKENS_PER_REQ:
                if cur_batch:
                    batches.append(cur_batch)
                    cur_batch, cur_tokens = [], 0
                batches.append([t])
                continue

            # Si agregarlo excede el límite -> cerrar lote
            if cur_tokens + tt > _MAX_TOKENS_PER_REQ and cur_batch:
                batches.append(cur_batch)
                cur_batch, cur_tokens = [], 0

            cur_batch.append(t)
            cur_tokens += tt

        if cur_batch:
            batches.append(cur_batch)
        return batches

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Embebe en lotes respetando el máximo de tokens por request."""
        vectors: List[List[float]] = []
        for batch in _make_batches(texts):
            resp = _client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=batch,
            )
            vectors.extend([d.embedding for d in resp.data])
        return vectors

else:
    # === Proveedor local (sentence-transformers) ===
    from sentence_transformers import SentenceTransformer

    _model_name = "intfloat/multilingual-e5-base"
    _model = SentenceTransformer(_model_name)

    def _prep_inputs(texts: List[str]) -> List[str]:
        return [f"passage: {t}" for t in texts]

    def embed_texts(texts: List[str]) -> List[List[float]]:
        return _model.encode(_prep_inputs(texts), normalize_embeddings=True).tolist()
