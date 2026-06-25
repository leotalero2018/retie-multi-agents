# app/agent/intent.py
"""Clasificación de intención de la consulta (TICKET-001 / H-901).

El camino de "respuesta exhaustiva" (más chunks recuperados, expansión por página,
4000 tokens de salida, salto del enriquecedor) antes se activaba SOLO con un regex
frágil (`todos`, `completa`, `sin omitir`…). La consulta canónica del negocio
—"dame los requerimientos para X"— no contiene ninguna de esas palabras y se
respondía con `TOP_K=4` / `MAX_TOKENS=600`, produciendo respuestas parciales sin
aviso.

Este módulo sustituye esa decisión por un clasificador de intención con salida
`{exhaustiva, tabla, puntual, smalltalk}`:

  1. Fast-path determinístico: smalltalk puro (saludos/cortesías) se resuelve por
     regex anclado — no se paga un LLM por "hola".
  2. Clasificador LLM barato (gpt-4o-mini) como FUENTE PRIMARIA del resto.
  3. Regex ampliado como FALLBACK robusto y siempre disponible (sin red, sin coste):
     cubre por sí solo el léxico normativo del negocio, de modo que el sistema sigue
     clasificando bien aunque el LLM esté desactivado o falle.

El resultado (`IntentResult`) trae además los flags `wants_table` / `wants_full` que
consume el grafo, y `route` (smalltalk | retrieve). La intención y su origen
(`regex` | `llm`) quedan registrados en la traza de Langfuse desde `route_entry`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from openai import OpenAI

from retie_agent.config import settings, as_bool

logger = logging.getLogger(__name__)

# Categorías válidas de intención.
_LABELS = ("exhaustiva", "tabla", "puntual", "smalltalk")


# ──────────────────────────────────────────────────────────────────────────────
# Léxico determinístico (fallback robusto, sin red)
# ──────────────────────────────────────────────────────────────────────────────

# Intención de TABLA por palabra clave.
_TABLE_RE = re.compile(r"\btablas?\b", re.IGNORECASE)


# Intención de respuesta EXHAUSTIVA. Además de los marcadores explícitos de
# completitud ("todos", "lista completa", "sin omitir", "numerales", "literales"),
# incluye el LÉXICO NORMATIVO DEL NEGOCIO (TICKET-001): pedir los requisitos /
# requerimientos / exigencias / condiciones / obligaciones de un tema, o preguntar
# "qué exige/requiere/establece/dice el artículo", debe activar el camino completo.
_FULL_RE = re.compile(
    r"\b("
    # — marcadores de completitud —
    r"todos?|todas?|completas?|completos?|sin\s+omitir|lista\s+completa|"
    r"al\s+pie\s+de\s+la\s+letra|literal(?:es)?|numerales?|cada\s+uno|exhaustiv[oa]s?|"
    # — léxico normativo del negocio —
    r"requerimientos?|requisitos?|exigencias?|"
    r"obligaci[oó]n(?:es)?|condici[oó]n(?:es)?|"
    # — "qué exige / requiere / establece / dice el artículo…" —
    r"qu[eé]\s+(?:exige|requiere|requieren|establece|establecen|dispone|determina|"
    r"se[ñn]ala|indica|contempla|pide|exigen|dice|deben?\s+cumplir|debe)"
    r")\b",
    re.IGNORECASE,
)


# Saludos / cortesías: se responden al instante, sin retrieval ni NotebookLM.
# El `[¡¿]*` inicial tolera signos de apertura ("¿quién eres?").
_SMALLTALK_RE = re.compile(
    r"^\s*[¡¿]*\s*(?:hola+|holi+|buen[oa]s(?:\s+(?:d[ií]as|tardes|noches))?|hey|hello|hi"
    r"|(?:muchas\s+)?gracias+|ok(?:ey)?|vale|listo|perfecto|genial|excelente"
    r"|adi[oó]s|hasta\s+luego|chao|nos\s+vemos"
    r"|qu[ié][eé]n\s+eres|qu[eé]\s+puedes\s+hacer|ayuda)\s*[!.?¡¿]*\s*$",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    """¿El texto trae una pregunta explícita (signo de interrogación)?

    Backstop NARROW del clasificador LLM (capa primaria): si el usuario escribió
    "?"/"¿" hay una pregunta real de por medio, así que el mensaje no puede ser
    smalltalk —aunque empiece con un saludo ("hola, ¿qué es el RETIE?")—. Se
    mantiene conservador a propósito (solo signos de interrogación) para no
    confundir un saludo largo y cortés con una consulta; esa decisión matizada
    (mezclas, imperativos sin signo) la toma el LLM con su prompt mejorado.
    """
    t = (text or "").strip()
    return "?" in t or "¿" in t


def _wants_table(text: str) -> bool:
    """Compat: True si el texto pide una tabla (regex determinístico)."""
    return bool(_TABLE_RE.search(text or ""))


def _wants_full(text: str) -> bool:
    """Compat: True si el texto pide una respuesta exhaustiva (regex determinístico)."""
    return bool(_FULL_RE.search(text or ""))


def _regex_intent(q: str) -> str:
    """Clasificación determinística por léxico. Prioridad:
    smalltalk → tabla → exhaustiva → puntual.

    `tabla` precede a `exhaustiva` porque una petición tabular ("la tabla completa
    220.55") debe enrutarse por table_node aunque contenga "completa".
    """
    if _SMALLTALK_RE.match(q):
        return "smalltalk"
    if _TABLE_RE.search(q):
        return "tabla"
    if _FULL_RE.search(q):
        return "exhaustiva"
    return "puntual"


# ──────────────────────────────────────────────────────────────────────────────
# Clasificador LLM barato (fuente primaria)
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Eres un clasificador de intención para un asistente de normativa eléctrica "
    "colombiana (RETIE y NTC 2050). Clasifica la consulta del usuario en EXACTAMENTE "
    "una de estas cuatro categorías y responde SOLO con esa palabra, sin explicar:\n\n"
    "- exhaustiva: pide el contenido normativo COMPLETO de un tema o artículo "
    "(requisitos, requerimientos, exigencias, condiciones, obligaciones; "
    "\"qué exige/requiere/establece/dice el artículo\"; \"todos los numerales\"; "
    "\"lista completa\"; \"sin omitir\").\n"
    "- tabla: pide una tabla concreta o datos tabulares "
    "(\"la tabla 220.55\", \"tabla de calibres\", \"ampacidades en tabla\").\n"
    "- puntual: pregunta por un dato único y acotado "
    "(\"¿cuál es la tensión nominal?\", \"¿qué significa GFCI?\", \"define acometida\").\n"
    "- smalltalk: SOLO saludo, agradecimiento o despedida a secas, o una pregunta "
    "sobre el propio bot (\"¿quién eres?\", \"¿qué puedes hacer?\").\n\n"
    "REGLA CLAVE: si el mensaje MEZCLA un saludo o cortesía CON una pregunta o "
    "petición de información (p. ej. \"hola, ¿qué es el RETIE?\"), clasifícalo por la "
    "PREGUNTA y NUNCA como smalltalk. Solo es smalltalk cuando NO hay ninguna "
    "pregunta ni petición sobre normativa eléctrica.\n\n"
    "Ejemplos:\n"
    "- \"hola\" → smalltalk\n"
    "- \"gracias, muy amable\" → smalltalk\n"
    "- \"buenas, ¿quién eres?\" → smalltalk\n"
    "- \"hola, ¿qué es el RETIE?\" → puntual\n"
    "- \"buenas tardes, dame los requisitos de puesta a tierra\" → exhaustiva\n"
    "- \"hey, pásame la tabla 220.55\" → tabla\n"
    "- \"¿cuál es la tensión nominal de servicio?\" → puntual\n\n"
    "Responde con una sola palabra: exhaustiva, tabla, puntual o smalltalk."
)

# Cliente OpenAI perezoso y compartido para el clasificador.
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    global _client
    if _client is None:
        try:
            _client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))
        except Exception:  # pragma: no cover - construcción del cliente
            return None
    return _client


def _parse_label(raw: str) -> Optional[str]:
    """Extrae una de las 4 etiquetas de la respuesta cruda del LLM (tolerante a
    comillas, mayúsculas o texto extra del tipo "Intención: exhaustiva")."""
    low = (raw or "").strip().lower()
    for label in _LABELS:
        if label in low:
            return label
    return None


def _llm_classify(question: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """Clasifica con el LLM barato. Devuelve una etiqueta válida o None (para que
    el llamador caiga al regex). Nunca lanza: cualquier fallo → None."""
    if not as_bool(getattr(settings, "INTENT_LLM_ENABLED", "true")):
        return None
    if not getattr(settings, "OPENAI_API_KEY", None):
        return None
    client = _get_client()
    if client is None:
        return None

    model = getattr(settings, "INTENT_MODEL", None) or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=8,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (question or "").strip()[:500]},
            ],
        )
        return _parse_label(resp.choices[0].message.content or "")
    except Exception as exc:
        logger.warning("intent LLM classifier failed, using regex fallback: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    intent: str          # exhaustiva | tabla | puntual | smalltalk
    source: str          # "regex" | "llm"
    route: str           # "smalltalk" | "retrieve"
    wants_table: bool
    wants_full: bool


def _build_result(intent: str, source: str) -> IntentResult:
    intent = intent if intent in _LABELS else "puntual"
    return IntentResult(
        intent=intent,
        source=source,
        route="smalltalk" if intent == "smalltalk" else "retrieve",
        wants_table=(intent == "tabla"),
        wants_full=(intent == "exhaustiva"),
    )


def classify_intent(
    question: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    is_media: bool = False,
) -> IntentResult:
    """Clasifica la intención de la consulta.

    Orden de decisión:
      1. smalltalk puro por regex anclado → respuesta inmediata, sin LLM.
      2. clasificador LLM (si está habilitado y disponible) → fuente primaria.
      3. regex ampliado → fallback determinístico (cubre el léxico de negocio).
      4. guardarraíl: una pregunta clara —o cualquier contenido de imagen/voz—
         nunca es smalltalk, aunque el LLM lo etiquete así.

    `is_media=True` indica que el texto proviene de una imagen (OCR/Vision) o de
    una nota de voz (transcripción); ese material nunca es un saludo.
    """
    q = (question or "").strip()

    # 1. Fast-path: smalltalk evidente no paga LLM. No aplica a contenido derivado
    #    de imagen/voz: el texto compuesto (OCR/Vision/transcripción) podría
    #    parecer un saludo, pero el usuario envió un medio para que lo procesemos.
    if not is_media and _SMALLTALK_RE.match(q):
        return _build_result("smalltalk", "regex")

    # 2. LLM como fuente primaria; 3. regex como fallback.
    llm_label = _llm_classify(q, history)
    if llm_label is not None:
        intent, source = llm_label, "llm"
    else:
        intent, source = _regex_intent(q), "regex"

    # 4. Guardarraíl determinístico (raíz de los bugs reportados): una consulta
    #    clara (con "?" o varias palabras) o cualquier contenido de imagen/voz
    #    JAMÁS es smalltalk. Si quedó etiquetado así, se reclasifica para que pase
    #    por recuperación en vez de devolver el mensaje de bienvenida.
    if intent == "smalltalk" and (is_media or _looks_like_question(q)):
        fallback = _regex_intent(q)
        intent = fallback if fallback != "smalltalk" else "puntual"
        source = f"{source}+guard"

    return _build_result(intent, source)
