# app/observability/obs.py
# Optional Langfuse observability with safe no-ops when disabled or when SDK API differs.

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
        from langfuse import Langfuse  # type: ignore
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
    When Langfuse is disabled/unavailable or the SDK API differs, yields None and does nothing.
    """
    lf = _get_client()
    if lf is None:
        yield None
        return

    trace_obj = None
    try:
        # Preferred (older SDKs): client.trace(...)
        if hasattr(lf, "trace"):
            trace_obj = lf.trace(name=name, user_id=user_id, metadata=metadata or {})  # type: ignore[attr-defined]
        # Newer SDKs may expose resource managers; if not present, just no-op
    except Exception:
        trace_obj = None

    try:
        yield trace_obj
    finally:
        try:
            if trace_obj is not None and hasattr(trace_obj, "end"):
                trace_obj.end()  # type: ignore[attr-defined]
        except Exception:
            pass


@contextmanager
def span_ctx(trace, name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Usage:
        with span_ctx(trace, "agent_answer"):
            ...
    No-op if trace is None or if the SDK doesn't provide span().
    """
    if trace is None:
        yield None
        return

    span_obj = None
    try:
        if hasattr(trace, "span"):
            span_obj = trace.span(name=name, metadata=metadata or {})  # type: ignore[attr-defined]
    except Exception:
        span_obj = None

    try:
        yield span_obj
    finally:
        try:
            if span_obj is not None and hasattr(span_obj, "end"):
                span_obj.end()  # type: ignore[attr-defined]
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
    Safe to call when observability is disabled or when the SDK API differs (no-op).
    """
    if trace is None:
        return
    try:
        if hasattr(trace, "generation"):
            gen = trace.generation(  # type: ignore[attr-defined]
                name=name,
                model=model or "",
                input=input_text,
                output=output_text,
                metadata=metadata or {},
                usage=usage or {},
            )
            try:
                if hasattr(gen, "end"):
                    gen.end()  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # Swallow metrics errors; never fail the request flow
        pass
