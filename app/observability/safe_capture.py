# app/observability/safe_capture.py
import os

DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", "true").lower() in ("1", "true", "yes")

def capture(*args, **kwargs):
    if DISABLE_TELEMETRY:
        return
    try:
        pass
    except Exception: 
        pass
