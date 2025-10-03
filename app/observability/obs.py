# app/observability/obs.py
# Optional Langfuse observability with safe no-ops when disabled or misconfigured.

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Any, Dict

from app.config import settings

# Cached client
_langfuse_client = None
_enabled_cache: Optional[bool] = None


def _enabled() -> bool:
    global _enabled_cache
    if _enabled_cache is not None:
        return _enabled_cache
    _enabled_cache = str(getattr(settings, "LANGFUSE_ENABLED", "false")).lower() in ("1", "true", "yes")
    return _enabled_cache


def _get_client():
    """Create (once) and return a Langfuse client, or None if disabled/unavailable."""
    global _langfuse_client
    if not _enabled():
        return None
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from langfuse import Langfuse  # import only if enabled
        _langfuse_client = Langfuse(
            public_key=getattr(settings, "LANGFUSE_PUBLIC_KEY", None),
            secret_key=getattr(settings, "LANGFUSE_SECRET_KEY", None),
            host=getattr(settings, "LANGFUSE_HOST", None),
        )
        return _langfuse_client
    except Exception:
        # Silently degrade to no-op
        return None


@contextmanager
def trace_ctx(name: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Usage:
        with trace_ctx("query_docs", user_id="api", metadata={"agent":"pymupdf"}) as trace:
            ...
    When Langfuse is disabled/unavailable, yields None and does nothing.
    """
    lf = _get_client()
    if lf is None:
        yield None
        return
    trace = lf.trace(name=name, user_id=user_id, metadata=metadata or {})
    try:
        yield trace
    finally:
        try:
            trace.end()
        except Exception:
            pass


@contextmanager
def span_ctx(trace, name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Usage:
        with span_ctx(trace, "agent_answer"):
            ...
    No-op if trace is None.
    """
    if trace is None:
        yield None
        return
    try:
        span = trace.span(name=name, metadata=metadata or {})
    except Exception:
        # degrade to no-op span
        yield None
        return

    try:
        yield span
    finally:
        try:
            span.end()
        except Exception:
            pass


def log_generation(
    trace,
    name: str,
    input_text: str,
    output_text: str,
    model: str = "",
    usage: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a single LLM generation attached to a trace.
    Safe to call when observability is disabled (no-op).
    """
    if trace is None:
        return
    try:
        gen = trace.generation(
            name=name,
            model=model or "",
            input=input_text,
            output=output_text,
            metadata=metadata or {},
            usage=usage or {},
        )
        try:
            gen.end()
        except Exception:
            pass
    except Exception:
        # Swallow metrics errors; never fail the request flow
        pass
