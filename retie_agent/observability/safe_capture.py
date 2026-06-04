# app/observability/safe_capture.py
# Minimal guard for optional telemetry hooks you might add later.

import os
from typing import Any

DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", "true").lower() in ("1", "true", "yes")

def capture(*_args: Any, **_kwargs: Any) -> None:
    """No-op unless you decide to wire custom telemetry here later."""
    if DISABLE_TELEMETRY:
        return
    try:
        # Place custom lightweight logging here if you need it.
        # Intentionally left blank to avoid side effects by default.
        return
    except Exception:
        # Never raise from telemetry
        return
