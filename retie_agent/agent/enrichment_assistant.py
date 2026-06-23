# app/agent/enrichment_assistant.py
import os
from openai import OpenAI
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Instrucciones del Assistant — VERSIONADAS EN EL REPO (TICKET-003 / H-903)
# ─────────────────────────────────────────────────────────────────────────────
# Decisión documentada: el enriquecimiento se MANTIENE pero se restringe a
# redacción conectiva. Tiene PROHIBIDO tocar citas, valores numéricos, tablas o
# listas, e introducir contenido externo a la evidencia. Esta es la fuente de
# verdad de lo que debe estar configurado en el Assistant de OpenAI; además se
# inyecta en cada mensaje (`enrich_response`) para que la restricción aplique
# aunque la config remota del Assistant difiera. Un verificador posterior
# (retie_agent/agent/fidelity.py) descarta cualquier salida que altere
# números / referencias / listas y entrega el borrador original.
ENRICHMENT_INSTRUCTIONS = (
    "Eres un editor de estilo para respuestas sobre normativa eléctrica colombiana "
    "(RETIE y NTC 2050) que YA están fundamentadas en fuentes recuperadas.\n"
    "Tu ÚNICA tarea es mejorar la LEGIBILIDAD y la prosa conectiva del borrador.\n\n"
    "TIENES ESTRICTAMENTE PROHIBIDO:\n"
    "  • cambiar, agregar o eliminar cualquier valor numérico, unidad, tensión, "
    "corriente, calibre, distancia, porcentaje o año;\n"
    "  • cambiar, agregar o eliminar cualquier referencia normativa "
    "(artículo, numeral, sección, número de tabla);\n"
    "  • agregar, eliminar, reordenar, fusionar o reformular los elementos de "
    "cualquier enumeración o lista (a), b), (C), 1., …);\n"
    "  • introducir hechos, requisitos o contenido que no estén en el borrador;\n"
    "  • resumir o acortar el borrador.\n\n"
    "SOLO PUEDES: corregir gramática, conectores, puntuación y el flujo de los "
    "párrafos para que el texto se lea con naturalidad. Conserva TODAS las citas, "
    "cifras y listas EXACTAMENTE como están. Mantén el mismo idioma del borrador."
)


class EnrichmentAssistant:
    def __init__(self, assistant_id: str, vector_store_id: Optional[str] = None):
        # Variable custom de Railway, env estándar, o settings (.env vía pydantic,
        # que NO exporta a os.environ — sin este fallback fallaba en local).
        from retie_agent.config import settings
        api_key = (
            os.getenv("Openai_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or settings.OPENAI_API_KEY
        )
        if not api_key:
            raise RuntimeError("Missing OpenAI API key. Please set Openai_API_KEY or OPENAI_API_KEY.")
        
        # pass it explicitly to the OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.assistant_id = assistant_id
        self.vector_store_id = vector_store_id

    def enrich_response(self, user_message: str, draft_response: str) -> str:
        input_prompt = (
            f"{ENRICHMENT_INSTRUCTIONS}\n\n"
            f"Pregunta del usuario: {user_message}\n\n"
            f"Borrador a pulir (consérvalo íntegro; solo mejora la redacción):\n"
            f"{draft_response}\n\n"
            f"Devuelve SOLO el texto mejorado, sin preámbulo, sin metacomentarios "
            f"(p. ej. 'aquí tienes una versión mejorada') y sin comillas alrededor — "
            f"tu salida va directo al usuario final."
        )

        thread = self.client.beta.threads.create()
        self.client.beta.threads.messages.create(thread_id=thread.id, role="user", content=input_prompt)
        self.client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=self.assistant_id)
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        return messages.data[0].content[0].text.value
