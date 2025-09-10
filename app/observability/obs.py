# app/observability/obs.py
#visualizacion del flujo del proyecto
import time
from contextlib import contextmanager
from typing import Optional
from app.config import settings

_langfuse = None

def get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    enabled = str(getattr(settings, "LANGFUSE_ENABLED", "false")).lower() == "true"
    if not enabled:
        return None
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=getattr(settings, "LANGFUSE_PUBLIC_KEY", None),
            secret_key=getattr(settings, "LANGFUSE_SECRET_KEY", None),
            host=getattr(settings, "LANGFUSE_HOST", None),
        )
        return _langfuse
    except Exception:
        return None


@contextmanager
def trace_ctx(name: str, user_id: Optional[str] = None, metadata: Optional[dict] = None):
    lf = get_langfuse()
    if lf is None:
        yield None
        return
    t = lf.trace(name=name, user_id=user_id, metadata=metadata or {})
    try:
        yield t
    finally:
        t.end()


@contextmanager
def span_ctx(trace, name: str, metadata: Optional[dict] = None):
    lf = get_langfuse()
    if lf is None or trace is None:
        yield None
        return
    s = trace.span(name=name, metadata=metadata or {})
    try:
        yield s
    finally:
        s.end()


def log_generation(trace, name: str, input_text: str, output_text: str, model: str = "", usage: Optional[dict] = None, metadata: Optional[dict] = None):
    lf = get_langfuse()
    if lf is None or trace is None:
        return
    gen = trace.generation(
        name=name,
        model=model or "",
        input=input_text,
        output=output_text,
        metadata=metadata or {},
        usage=usage or {},  
    )
    gen.end()
