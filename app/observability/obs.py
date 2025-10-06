# app/observability/obs.py
# Langfuse v3 (OTel-based) helpers with safe no-ops when disabled.

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Any, Dict

from app.config import settings

_langfuse = None
_enabled_cache: Optional[bool] = None


def _enabled() -> bool:
    global _enabled_cache
    if _enabled_cache is not None:
        return _enabled_cache
    _enabled_cache = str(getattr(settings, "LANGFUSE_ENABLED", "false")).lower() in ("1", "true", "yes")
    return _enabled_cache


def _get_client():
    """
    Return the global v3 client (or None if disabled/unavailable).
    v3 best practice is get_client(); env vars must be set.
    """
    if not _enabled():
        return None
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        # v3 style
        from langfuse import get_client  # type: ignore
        _langfuse = get_client()  # initialized from env (PUBLIC_KEY, SECRET_KEY, HOST)
        return _langfuse
    except Exception:
        return None


@contextmanager
def trace_ctx(name: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Starts a root span; in v3 this implicitly creates the trace.
    """
    lf = _get_client()
    if lf is None:
        yield None
        return

    try:
        with lf.start_as_current_span(name=name) as span:
            # set user_id / tags on the current trace
            try:
                if user_id:
                    lf.update_current_trace(user_id=user_id)
                if metadata:
                    # You can also pass tags via metadata.get("tags", [...])
                    lf.update_current_span(input=metadata)  # light enrichment
            except Exception:
                pass
            yield span
    except Exception:
        # degrade to no-op
        yield None


@contextmanager
def span_ctx(trace, name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Creates a child span under the currently active one (trace arg is unused but kept for API parity).
    """
    lf = _get_client()
    if lf is None:
        yield None
        return

    try:
        with lf.start_as_current_span(name=name) as span:
            try:
                if metadata:
                    lf.update_current_span(input=metadata)
            except Exception:
                pass
            yield span
    except Exception:
        yield None


def log_generation(
    trace,  # kept for API parity; unused in v3 helpers
    name: str,
    input_text: str,
    output_text: str,
    model: str = "",
    usage: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Records one LLM call as a Generation nested under the current span.
    Safe no-op when disabled/unavailable.
    """
    lf = _get_client()
    if lf is None:
        return
    try:
        with lf.start_as_current_generation(
            name=name,
            model=model or "",
            input=input_text if isinstance(input_text, (str, bytes)) else str(input_text),
        ) as gen:
            try:
                gen.update(
                    output=output_text,
                    usage_details=usage or {},
                    metadata=metadata or {},
                )
            except Exception:
                pass
    except Exception:
        pass
