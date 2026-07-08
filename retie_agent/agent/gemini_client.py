"""Cliente de Gemini File Search — RAG gestionado oficial del API de Gemini.

Espeja el contrato de NotebookLMClient.ask_question: devuelve `(texto, fuentes)`
para que gemini_node encaje en el mismo patrón de fan-out que notebooklm_node,
sin tocar la mecánica del grafo.

A diferencia de NotebookLM (navegador + cookies que caducan), File Search es una
API con key: sin sesión, sin login, con citas a nivel de página. Ver
docs: https://ai.google.dev/gemini-api/docs/file-search

SPIKE (Fase 1a de la propuesta de modernización): el objetivo es validar CALIDAD
de respuestas Gemini vs NotebookLM sobre el mismo golden set (TICKET-005), no
reemplazar nada todavía. El cliente degrada a "" si no hay SDK o GEMINI_API_KEY,
para que el pipeline actual nunca se rompa por su ausencia.

Requisito: pip install google-genai
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Gemini responde 503 UNAVAILABLE ("high demand", transitorio) o 429 (rate limit)
# bajo carga. Son reintentables con backoff; el resto de errores no.
_RETRYABLE_MARKERS = ("503", "unavailable", "429", "resource_exhausted", "high demand")
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # segundos: 2, 4, 8


class GeminiError(RuntimeError):
    """Fallo recuperable del cliente de Gemini File Search."""


class GeminiFileSearchClient:
    """Consulta un File Search Store de Gemini y devuelve respuesta + citas.

    Uso:
        client = GeminiFileSearchClient(api_key=..., model=..., store=...)
        answer, sources = client.ask_question("¿Qué exige el RETIE sobre...?")
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gemini-2.5-flash",
        store: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.store = (store or "").strip()
        self.timeout = timeout
        self._client = None
        self._lock = threading.Lock()

    # ── SDK lazy ──────────────────────────────────────────────────────────────
    def _ensure_client(self):
        """Instancia el SDK de Google GenAI la primera vez (import perezoso).

        Perezoso a propósito: importar google.genai en el import del módulo rompía
        el arranque en entornos sin la dependencia (tests, pipeline offline)."""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            if not self.api_key:
                raise GeminiError("GEMINI_API_KEY no configurada.")
            try:
                from google import genai  # type: ignore
            except Exception as exc:  # pragma: no cover - depende del entorno
                raise GeminiError(
                    "SDK google-genai no instalado. Ejecuta: pip install google-genai"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
            return self._client

    def is_available(self) -> bool:
        """True si hay key + store configurados (sin llamar a la red)."""
        return bool(self.api_key and self.store)

    # ── Consulta ──────────────────────────────────────────────────────────────
    def ask_question(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Responde `query` con grounding sobre el File Search Store.

        Devuelve (texto_respuesta, fuentes) donde cada fuente es
        {source, page, snippet} — el mismo shape que consumen stylist_node y el
        formateador de fuentes del grafo. Lanza GeminiError si algo falla."""
        if not self.store:
            raise GeminiError("GEMINI_FILE_SEARCH_STORE no configurado.")

        client = self._ensure_client()
        from google.genai import types  # type: ignore

        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[self.store]
                    )
                )
            ],
        )

        # Reintento con backoff ante 503/429 (saturación temporal de Gemini).
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=query,
                    config=config,
                )
                answer = (getattr(resp, "text", None) or "").strip()
                sources = self._extract_citations(resp)
                return answer, sources
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                retryable = any(m in msg for m in _RETRYABLE_MARKERS)
                if not retryable or attempt == _MAX_RETRIES:
                    break
                wait = _BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Gemini 503/429 (intento %d/%d) — reintentando en %.0fs",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)

        raise GeminiError(f"Gemini File Search falló: {last_exc}") from last_exc

    @staticmethod
    def _extract_citations(resp: Any) -> List[Dict[str, Any]]:
        """Best-effort: extrae las citas de grounding_metadata del response.

        El shape del SDK puede variar entre versiones; se envuelve todo en
        try/except y se devuelve [] si no se encuentra nada (nunca rompe)."""
        sources: List[Dict[str, Any]] = []
        try:
            candidates = getattr(resp, "candidates", None) or []
            for cand in candidates:
                gm = getattr(cand, "grounding_metadata", None)
                if gm is None:
                    continue
                chunks = getattr(gm, "grounding_chunks", None) or []
                for ch in chunks:
                    ctx = getattr(ch, "retrieved_context", None) or getattr(ch, "web", None)
                    if ctx is None:
                        continue
                    # En File Search el título suele venir vacío; document_name trae
                    # el nombre del PDF subido, que es la cita útil para el usuario.
                    src = (
                        getattr(ctx, "title", None)
                        or getattr(ctx, "document_name", None)
                        or getattr(ctx, "uri", None)
                        or "Gemini File Search"
                    )
                    sources.append(
                        {
                            "source": src,
                            "page": getattr(ctx, "page_number", None),
                            "snippet": (getattr(ctx, "text", None) or "")[:300],
                        }
                    )
        except Exception:  # pragma: no cover - defensivo
            pass
        return sources
