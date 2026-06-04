# app/observability/obs.py
# Langfuse v3 helpers (+ support for custom observation types like "retriever", "chain", "agent")

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Any, Dict
import os

from retie_agent.config import settings

_langfuse = None
_enabled_cache: Optional[bool] = None


def _enabled() -> bool:
    global _enabled_cache
    if _enabled_cache is not None:
        return _enabled_cache
    _enabled_cache = str(getattr(settings, "LANGFUSE_ENABLED", "false")).lower() in ("1", "true", "yes")
    return _enabled_cache


def _get_client():
    """Return the global v3 client (or None if disabled/unavailable)."""
    if not _enabled():
        return None
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        from langfuse import Langfuse

        pk = getattr(settings, "LANGFUSE_PUBLIC_KEY", None) or os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = getattr(settings, "LANGFUSE_SECRET_KEY", None) or os.getenv("LANGFUSE_SECRET_KEY")
        host = (
            getattr(settings, "LANGFUSE_HOST", None)
            or os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
        )

        kwargs: dict = {}
        if pk:
            kwargs["public_key"] = pk
        if sk:
            kwargs["secret_key"] = sk
        if host:
            kwargs["host"] = host

        _langfuse = Langfuse(**kwargs)
        return _langfuse
    except Exception:
        return None


@contextmanager
def trace_ctx(
    name: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_input: Optional[Dict[str, Any]] = None,
):
    """
    Root span; set overall Input on the root (question, agent, etc.).
    """
    lf = _get_client()
    if lf is None:
        yield None
        return

    # Setup before yield — exceptions here are safe to catch.
    try:
        ctx_mgr = lf.start_as_current_span(name=name)
    except Exception:
        yield None
        return

    with ctx_mgr as span:
        try:
            if user_id:
                lf.update_current_trace(user_id=user_id)
            if trace_input:
                lf.update_current_span(input=trace_input)
            if metadata:
                for k, v in metadata.items():
                    span.set_attribute(f"meta.{k}", v)
        except Exception:
            pass
        # yield is outside any try/except so exceptions thrown by the caller
        # propagate normally and don't trigger "generator didn't stop after throw()"
        yield span


@contextmanager
def span_ctx(
    trace,                      # kept for API parity
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    as_type: Optional[str] = None,     # <-- important for Agent Graph
    span_input: Optional[Dict[str, Any]] = None,
):
    """
    Child observation. If `as_type` is provided (e.g., "retriever", "chain", "agent"),
    Langfuse will classify this observation accordingly, enabling the Agent Graph view.
    """
    lf = _get_client()
    if lf is None:
        yield None
        return

    # Setup before yield — exceptions here are safe to catch.
    try:
        try:
            ctx_mgr = lf.start_as_current_span(name=name, as_type=as_type)  # type: ignore[arg-type]
        except TypeError:
            ctx_mgr = lf.start_as_current_span(name=name)
    except Exception:
        yield None
        return

    with ctx_mgr as span:
        try:
            if span_input:
                lf.update_current_span(input=span_input)
            if metadata:
                for k, v in metadata.items():
                    span.set_attribute(f"meta.{k}", v)
        except Exception:
            pass
        # yield is outside any try/except so exceptions thrown by the caller
        # propagate normally and don't trigger "generator didn't stop after throw()"
        yield span


def log_generation(
    trace,  # unused (API parity)
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
