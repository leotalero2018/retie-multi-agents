# app/ingestion/embedder.py
from typing import List
from retie_agent.config import settings

# The retriever.search() will call embed_texts() and then use query_embeddings
# against Chroma (no EF bound to the collection). Make sure this uses THE SAME
# model used at indexing time. For 1536-dim vectors use "text-embedding-3-small"
# (or "text-embedding-ada-002" if you indexed long ago with the legacy model).

if getattr(settings, "EMBEDDING_PROVIDER", "openai").lower() == "openai":
    import logging
    import os
    import httpx
    from openai import OpenAI, AuthenticationError

    _log = logging.getLogger(__name__)

    def _candidate_keys() -> List[str]:
        """Keys candidatas en orden de preferencia, sin duplicados.

        CHROMA_OPENAI_API_KEY primero (convención del ecosistema Chroma), pero si
        está vencida embed_texts() cae automáticamente a OPENAI_API_KEY en vez de
        fallar con 401 (caso real: key vieja olvidada en el entorno).
        """
        keys: List[str] = []
        for k in (
            os.getenv("CHROMA_OPENAI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            settings.OPENAI_API_KEY,
        ):
            if k and k not in keys:
                keys.append(k)
        if not keys:
            raise RuntimeError(
                "Missing OpenAI API key. Set CHROMA_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        return keys

    def _build_openai_client(api_key: str) -> OpenAI:
        # Optional proxy support
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
        http_client = httpx.Client(proxies=proxy, timeout=60) if proxy else None
        # Base URL allows Azure/OpenRouter/etc if needed
        base_url = os.getenv("OPENAI_BASE_URL") or None
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    _client: OpenAI | None = None
    _key_index = 0

    # Default to 1536-dim model unless explicitly overridden in settings
    _MODEL = getattr(settings, "EMBEDDING_MODEL", None) or "text-embedding-3-small"

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the configured OpenAI embedding model.
        IMPORTANT:
          - Do NOT prepend prefixes like 'query:' or 'passage:' for OpenAI models.
          - Keep one vector per input in the returned list.
        Si la key activa devuelve 401, reintenta con la siguiente disponible
        (CHROMA_OPENAI_API_KEY → OPENAI_API_KEY) y la deja fijada.
        """
        global _client, _key_index
        if not texts:
            return []

        keys = _candidate_keys()
        _key_index = min(_key_index, len(keys) - 1)
        while True:
            if _client is None:
                _client = _build_openai_client(keys[_key_index])
            try:
                resp = _client.embeddings.create(model=_MODEL, input=texts)
                return [d.embedding for d in resp.data]
            except AuthenticationError:
                if _key_index >= len(keys) - 1:
                    raise
                _log.warning(
                    "OpenAI embeddings: key #%d inválida (401); probando la siguiente",
                    _key_index + 1,
                )
                _key_index += 1
                _client = None

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
