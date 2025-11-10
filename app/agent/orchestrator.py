# app/agent/orchestrator.py
# Backward-compatible shim to preserve imports elsewhere.
from typing import Optional
from .retie_agent import answer_question as _answer_question, _dedupe_hits as _dedupe_hits

# re-export for compatibility
answer_question = _answer_question
_dedupe_hits = _dedupe_hits

from app.agent.enrichment_assistant import EnrichmentAssistant

# Initialize once (could be global or dependency-injected)
enrichment_agent = EnrichmentAssistant(
    assistant_id="asst_qthy1ZfTpr2ps0mruX30zVlc",
    vector_store_id="vs_69050fe6e43c8191be28bac47c3f565f"
)

async def handle_user_query(user_message: str):
    # Step 1: Primary agent produces base response
    draft_response = await primary_agent.respond(user_message)

    # Step 2: Enrich it using OpenAI Assistant
    enriched_response = enrichment_agent.enrich_response(user_message, draft_response)

    return enriched_response
