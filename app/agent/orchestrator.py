# app/agent/orchestrator.py
# Backward-compatible shim to preserve imports elsewhere.
from typing import Optional
from .retie_agent import answer_question as _answer_question, _dedupe_hits as _dedupe_hits

# re-export for compatibility
answer_question = _answer_question
_dedupe_hits = _dedupe_hits
