from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Any, Dict
import logging
import os

from retie_agent.config import settings

log = logging.getLogger(__name__)

_langfuse = None
_enabled_cache: Optional[bool] = None


def _enabled() -> bool:
    global _enabled_cache
    if _enabled_cache is not None:
        return _enabled_cache
    _enabled_cache = str(getattr(settings, "LANGFUSE_ENABLED", "false")).lower() in ("1", "true", "yes")
    return _enabled_cache


def _get_client():
    if not _enabled():
        return None
    global _langfuse, _enabled_cache
    if _langfuse is not None:
        return _langfuse
    try:
        from langfuse import Langfuse

        pk_raw = getattr(settings, "LANGFUSE_PUBLIC_KEY", None) or os.getenv("LANGFUSE_PUBLIC_KEY")
        sk_raw = getattr(settings, "LANGFUSE_SECRET_KEY", None) or os.getenv("LANGFUSE_SECRET_KEY")
        host_raw = (
            getattr(settings, "LANGFUSE_HOST", None)
            or os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
        )

        # Railway/os.environ no quita comillas ni saltos de línea como sí hace .env.
        # Un '\n' o comilla al final del secret corrompe el header Basic auth → 401.
        def _clean(v):
            if v is None:
                return None
            return v.strip().strip('"').strip("'").strip()

        pk = _clean(pk_raw)
        sk = _clean(sk_raw)
        host = _clean(host_raw)

        # Diagnóstico: revela caracteres ocultos (saltos de línea, comillas, espacios)
        for label, raw, cleaned in (("pk", pk_raw, pk), ("sk", sk_raw, sk), ("host", host_raw, host)):
            if raw is not None and len(raw) != len(cleaned or ""):
                log.warning(
                    "[Langfuse] %s tenía caracteres extra: len %d -> %d. "
                    "raw_repr_tail=%r (corregido automáticamente)",
                    label, len(raw), len(cleaned or ""), raw[-4:],
                )

        if not pk or not sk:
            log.warning(
                "[Langfuse] LANGFUSE_ENABLED=true pero faltan credenciales — "
                "public_key=%s secret_key=%s host=%s. "
                "Configura las variables en Railway (o .env local).",
                "OK" if pk else "MISSING",
                "OK" if sk else "MISSING",
                host or "MISSING",
            )
            _enabled_cache = False
            return None

        otlp_host = host or "https://cloud.langfuse.com"

        log.info(
            "[Langfuse] Inicializando cliente — pk=%s... host=%s",
            pk[:8], otlp_host,
        )

        kwargs: dict = {"public_key": pk, "secret_key": sk}
        if host:
            # base_url tiene la máxima precedencia en el SDK; lo pasamos explícito
            # para que un LANGFUSE_BASE_URL externo no nos mande a otra región.
            kwargs["host"] = host
            kwargs["base_url"] = host

        _langfuse = Langfuse(**kwargs)

        try:
            _langfuse.auth_check()
            log.info("[Langfuse] auth_check OK — trazas activas")
        except Exception as e:
            log.warning(
                "[Langfuse] auth_check falló — Langfuse deshabilitado permanentemente. "
                "Verifica LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY y LANGFUSE_HOST en Railway. "
                "Mensaje del servidor: %s",
                getattr(e, "body", str(e)),
            )
            try:
                _langfuse.shutdown()
            except Exception:
                pass
            _langfuse = None
            _enabled_cache = False  # evita bucle de re-inicialización y nuevos spans OTEL
            return None

        return _langfuse
    except Exception as exc:
        log.warning("[Langfuse] Error al inicializar cliente: %s", exc)
        _enabled_cache = False
        return None


@contextmanager
def trace_ctx(
    name: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_input: Optional[Dict[str, Any]] = None,
):
    """Root observation — groups all child spans under one trace."""
    lf = _get_client()
    if lf is None:
        yield None
        return

    meta = {**(metadata or {})}
    if user_id:
        meta["user_id"] = user_id

    try:
        ctx_mgr = lf.start_as_current_observation(
            name=name,
            as_type="span",
            input=trace_input,
            metadata=meta if meta else None,
        )
    except Exception:
        yield None
        return

    with ctx_mgr as span:
        try:
            if trace_input:
                lf.set_current_trace_io(input=trace_input)
        except Exception:
            pass
        yield span


@contextmanager
def span_ctx(
    trace,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    as_type: Optional[str] = None,
    span_input: Optional[Dict[str, Any]] = None,
):
    """Child observation. as_type controls the Langfuse graph icon."""
    lf = _get_client()
    if lf is None:
        yield None
        return

    try:
        ctx_mgr = lf.start_as_current_observation(
            name=name,
            as_type=as_type or "span",
            input=span_input,
            metadata=metadata if metadata else None,
        )
    except Exception:
        yield None
        return

    with ctx_mgr as span:
        yield span


def log_generation(
    trace,
    name: str,
    input_text: str,
    output_text: str,
    model: str = "",
    usage: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one LLM call as a Generation under the current span."""
    lf = _get_client()
    if lf is None:
        return
    try:
        with lf.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model or "",
            input=input_text if isinstance(input_text, (str, bytes)) else str(input_text),
            metadata=metadata if metadata else None,
        ) as gen:
            try:
                gen.update(
                    output=output_text,
                    usage_details=usage or {},
                )
            except Exception:
                pass
    except Exception:
        pass
