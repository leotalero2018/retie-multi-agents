# app/ingestion/embedder.py
from typing import List
from app.config import settings

# The retriever.search() will call embed_texts() and then use query_embeddings
# against Chroma (no EF bound to the collection). Make sure this uses THE SAME
# model used at indexing time. For 1536-dim vectors use "text-embedding-3-small"
# (or "text-embedding-ada-002" if you indexed long ago with the legacy model).

if getattr(settings, "EMBEDDING_PROVIDER", "openai").lower() == "openai":
    import os
    import httpx
    from openai import OpenAI

    def _build_openai_client() -> OpenAI:
        # Optional proxy support
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
        http_client = httpx.Client(proxies=proxy, timeout=60) if proxy else None

        # Base URL allows Azure/OpenRouter/etc if needed
        base_url = os.getenv("OPENAI_BASE_URL") or None

        # Prefer CHROMA_OPENAI_API_KEY if present (Chroma ecosystem convention),
        # otherwise fall back to OPENAI_API_KEY from settings/env.
        api_key = (
            os.getenv("CHROMA_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or settings.OPENAI_API_KEY
        )

        if not api_key:
            raise RuntimeError(
                "Missing OpenAI API key. Set CHROMA_OPENAI_API_KEY or OPENAI_API_KEY."
            )

        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    _client = _build_openai_client()

    # Default to 1536-dim model unless explicitly overridden in settings
    _MODEL = getattr(settings, "EMBEDDING_MODEL", None) or "text-embedding-3-small"

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the configured OpenAI embedding model.
        IMPORTANT:
          - Do NOT prepend prefixes like 'query:' or 'passage:' for OpenAI models.
          - Keep one vector per input in the returned list.
        """
        if not texts:
            return []

        # OpenAI API supports batching; send as a single request unless extremely large.
        resp = _client.embeddings.create(model=_MODEL, input=texts)
        return [d.embedding for d in resp.data]

else:
    # Local provider via sentence-transformers (keep as a dev fallback).
    from sentence_transformers import SentenceTransformer

    # NOTE: Local model dimension must match the one used at index time if mixing!
    # For consistency with your current index, prefer OpenAI in production.
    _MODEL_NAME = "intfloat/multilingual-e5-base"
    _model = SentenceTransformer(_MODEL_NAME)

    def _prep_inputs(texts: List[str]) -> List[str]:
        # E5 family expects prefixes like "query:" / "passage:", but since your
        # index uses OpenAI 1536-dim, avoid mixing providers in production.
        return [t for t in texts]

    def embed_texts(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = _model.encode(_prep_inputs(texts), normalize_embeddings=True)
        return vecs.tolist()
