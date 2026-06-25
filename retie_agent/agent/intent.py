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
#   ambiguous = fragmento vago/incompleto ("que", "que es") → se pide aclaración.
_LABELS = ("exhaustiva", "tabla", "puntual", "smalltalk", "ambiguous")


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
# `_GREET_CLAUSE` es UNA cláusula de saludo; el patrón completo admite VARIAS
# encadenadas ("hola buenas tardes", "hola, ¿cómo estás?") pero NO fragmentos
# sueltos como "que" o "que es" (esos caen en `ambiguous`). El regex es la ÚNICA
# autoridad para emitir el saludo: el LLM no puede, por sí solo, disparar el
# mensaje de bienvenida (evita falsos positivos del tipo "que" → saludo).
_GREET_CLAUSE = (
    r"(?:hola+|holi+|holis+|buen[oa]s(?:\s+(?:d[ií]as|tardes|noches))?"
    r"|hey+|hello+|hi+|saludos|qu[eé]\s+tal|qu[eé]\s+m[aá]s|c[oó]mo\s+est[aá]s"
    r"|(?:muchas\s+)?gracias+|ok(?:ey)?|vale|listo|perfecto|genial|excelente|de\s+nada"
    r"|adi[oó]s|hasta\s+luego|cha[uo]+|nos\s+vemos"
    r"|qu[ié][eé]n\s+eres|qu[eé]\s+puedes\s+hacer|ayuda)"
)
_SMALLTALK_RE = re.compile(
    r"^\s*[¡¿]*\s*"
    + _GREET_CLAUSE
    + r"(?:[\s,.!?¡¿]+(?:y\s+)?" + _GREET_CLAUSE + r")*"
    + r"\s*[!.?¡¿]*\s*$",
    re.IGNORECASE,
)


# Palabras funcionales/interrogativas que no aportan tema. Un mensaje compuesto
# SOLO por ellas es demasiado vago para recuperar algo útil ("que", "que es").
_STOPWORDS = {
    "que", "qué", "es", "son", "el", "la", "los", "las", "un", "una", "unos",
    "unas", "de", "del", "al", "en", "con", "por", "para", "se", "su", "sus",
    "lo", "le", "me", "mi", "tu", "te", "como", "cómo", "cual", "cuál", "cuales",
    "cuáles", "eso", "esa", "ese", "esto", "esta", "este", "asi", "así", "mas",
    "más", "muy", "ya", "hay", "sobre", "cuanto", "cuánto", "cuando", "cuándo",
}


def _is_too_vague(text: str) -> bool:
    """¿El mensaje es demasiado corto/incompleto para recuperar algo útil?

    True cuando, quitando palabras funcionales/interrogativas, no queda ningún
    término de contenido: "que", "que es", "y eso", "cómo así". No es un saludo
    ni una consulta respondible → conviene pedir aclaración en vez de saludar.
    """
    content = [
        w for w in re.findall(r"\w+", (text or "").lower(), re.UNICODE)
        if len(w) > 1 and w not in _STOPWORDS
    ]
    return not content


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
    intent: str          # exhaustiva | tabla | puntual | smalltalk | ambiguous
    source: str          # "regex" | "llm"
    route: str           # "smalltalk" | "ambiguous" | "retrieve"
    wants_table: bool
    wants_full: bool


def _build_result(intent: str, source: str) -> IntentResult:
    intent = intent if intent in _LABELS else "puntual"
    if intent == "smalltalk":
        route = "smalltalk"
    elif intent == "ambiguous":
        route = "ambiguous"
    else:
        route = "retrieve"
    return IntentResult(
        intent=intent,
        source=source,
        route=route,
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

    El SALUDO solo se emite si el mensaje se IDENTIFICA como saludo (regex anclado);
    no hay otra vía. Orden de decisión:
      1. Saludo/cortesía CONFIRMADO por regex anclado → smalltalk. Única puerta al
         mensaje de bienvenida. No aplica a imagen/voz.
      2. Fragmento vago/incompleto ("que", "que es") y SIN historial que lo
         complete → ambiguous (se pide aclaración; ni saludo ni "sin evidencia").
      3. Clasificador LLM (si está habilitado) → fuente primaria del resto.
      4. Regex ampliado → fallback determinístico (léxico de negocio).
      5. Guardarraíl: a esta altura el regex NO confirmó saludo, así que cualquier
         'smalltalk' del LLM es un falso positivo → se enruta a recuperación.

    `is_media=True` indica texto derivado de imagen (OCR/Vision) o voz; nunca es
    saludo ni se trata como vago.
    """
    q = (question or "").strip()

    # 1. Saludo confirmado por regex anclado → único camino al mensaje de bienvenida.
    #    No aplica a imagen/voz: el usuario envió un medio para que lo procesemos.
    if not is_media and _SMALLTALK_RE.match(q):
        return _build_result("smalltalk", "regex")

    # 2. Fragmento vago/incompleto (sin historial previo que lo complete) → aclarar.
    if not is_media and not history and _is_too_vague(q):
        return _build_result("ambiguous", "regex")

    # 3. LLM como fuente primaria; 4. regex como fallback.
    llm_label = _llm_classify(q, history)
    if llm_label is not None:
        intent, source = llm_label, "llm"
    else:
        intent, source = _regex_intent(q), "regex"

    # 5. Guardarraíl: el saludo solo lo decide el regex anclado (paso 1). Si el LLM
    #    etiquetó smalltalk pero el regex no lo confirmó, es un falso positivo
    #    ("que", "explícame", "info") → se reclasifica hacia recuperación.
    if intent == "smalltalk":
        fallback = _regex_intent(q)
        intent = fallback if fallback != "smalltalk" else "puntual"
        source = f"{source}+guard"

    return _build_result(intent, source)
