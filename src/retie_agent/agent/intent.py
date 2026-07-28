# app/agent/intent.py
"""Clasificación de intención de la consulta — Intent v3 (classifier_node).

Clasificador ÚNICO del grafo (el v1 de una palabra fue retirado; v3 es el
definitivo). Produce un `IntentResultV3` estructurado que gobierna al deep agent:
taxonomía `{puntual, exhaustiva, tabla, comparativa, procedimiento, verificacion,
fuera_de_dominio, smalltalk}` (+ `ambiguous` vía regex), formato y longitud de
entrega, entidades normativas y confianza.

Orden de decisión:
  1. Fast-path determinístico: smalltalk puro (saludos/cortesías) se resuelve por
     regex anclado — no se paga un LLM por "hola". Única puerta al saludo.
  2. Fragmento vago sin historial → ambiguous (se pide aclaración).
  3. Clasificador LLM con salida JSON (structured output) como fuente primaria,
     con CASCADA: los casos dudosos re-clasifican con el modelo potente
     (INTENT_V3_ESCALATION_MODEL).
  4. Regex ampliado como FALLBACK robusto y siempre disponible (sin red, sin
     coste): cubre por sí solo el léxico normativo del negocio (H-901), de modo
     que el sistema sigue clasificando bien aunque el LLM esté caído.
  5. Guardarraíles: smalltalk del LLM sin confirmación del regex → falso
     positivo; fuera_de_dominio solo corta el pipeline con confianza alta.

Los flags calientes `wants_table` / `wants_full` se derivan por REGLA DURA para
los nodos existentes; el resto viaja en `GraphState['intent_meta']` y queda
registrado en la traza de Langfuse desde `classifier_node`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple

from openai import OpenAI

from retie_agent.config import settings, as_bool
from retie_agent.llm.provider import create_chat_completion

logger = logging.getLogger(__name__)

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


# Alargamientos expresivos ("graciaaas", "holaaaa", "heyyy"): el regex anclado
# solo tolera repetir la ÚLTIMA letra de cada saludo ("gracias+"), así que un
# alargamiento interno lo sacaba del fast-path y el guardarraíl 5a terminaba
# mandando un "gracias" al RAG completo. Colapsar rachas de 3+ letras iguales a
# una sola es seguro en español (no hay palabras legítimas con triples letras);
# 2 repeticiones se respetan ("ll", "rr", "cc").
_ELONGATION_RE = re.compile(r"(\w)\1{2,}", re.UNICODE)


def _collapse_elongations(text: str) -> str:
    """Normaliza alargamientos expresivos SOLO para los regex de smalltalk."""
    return _ELONGATION_RE.sub(r"\1", text or "")


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


# ══════════════════════════════════════════════════════════════════════════════
# Intent v3 (classifier_node — FINAL_IMPLEMENTATION_NODES §4)
#
# Salida estructurada que gobierna al deep agent: taxonomía ampliada
# (comparativa/procedimiento/verificacion/fuera_de_dominio), formato y longitud
# de entrega, entidades normativas extraídas por regex y cascada de modelos
# (nano primero; los casos dudosos escalan al modelo potente). Conserva TODOS
# los guardarraíles de v1: el saludo solo lo emite el regex anclado, la vaguedad
# se corta antes del LLM, y fuera_de_dominio jamás corta el pipeline sin
# confianza alta del LLM.
# ══════════════════════════════════════════════════════════════════════════════

_V3_INTENTS = (
    "smalltalk", "puntual", "exhaustiva", "tabla", "comparativa",
    "procedimiento", "verificacion", "fuera_de_dominio",
)
_V3_FORMATS = ("markdown", "markdown_table", "bullet_list", "comparison", "steps")
_V3_LENGTHS = ("short", "medium", "detailed")
_V3_COMPLEXITY = ("low", "medium", "high")

# Intents donde equivocarse es más caro (cortan el RAG o cambian el procedimiento
# del deep agent): escalan al modelo potente con un umbral más exigente.
_V3_SENSITIVE_INTENTS = ("fuera_de_dominio", "comparativa", "verificacion")

# Comparativa por léxico (fallback determinístico, sin red).
_COMPARE_RE = re.compile(
    r"\b(diferencias?|comparar?|comparaci[oó]n|versus|vs\.?|frente\s+a|"
    r"cu[aá]l\s+es\s+mejor|en\s+qu[eé]\s+se\s+diferencian?)\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Extracción de entidades normativas (determinística, corre SIEMPRE)
# ──────────────────────────────────────────────────────────────────────────────

# Los IDs de tabla llevan sufijo de letra en RETIE ("2.3.26.2.2.1.a") y NTC
# ("392.10(A)" / "392.10 (A)"): capturarlo completo es requisito del lookup
# canónico por identificador. El sufijo ".a" se exige en minúscula ((?-i:...))
# para no absorber la inicial de un título ("tabla 220.55. Factores").
_TABLE_ID_RE = r"(\d+(?:[.\-]\d+)*(?:\.(?-i:[a-zñ])\b|\s*\((?-i:[A-Za-zñ])\))?)"

_ENTITY_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("tabla",    re.compile(r"\btablas?\s+" + _TABLE_ID_RE, re.IGNORECASE)),
    ("articulo", re.compile(r"\bart[ií]culos?\s+" + _TABLE_ID_RE + r"[°ºo]?", re.IGNORECASE)),
    ("seccion",  re.compile(r"\bsecci[oó]n(?:es)?\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)),
    ("capitulo", re.compile(r"\bcap[ií]tulos?\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)),
    ("anexo",    re.compile(r"\banexos?\s+([a-z0-9]+)\b", re.IGNORECASE)),
    ("norma",    re.compile(r"\b(ntc\s*\d+(?:-\d+)?|retie|retilap)\b", re.IGNORECASE)),
    # Referencias numéricas sueltas tipo NEC/NTC: "la 220.55", "el 110-14".
    # DEBE ir de última: solo se emite si tabla/articulo/seccion no la capturaron ya.
    ("ref",      re.compile(r"\b(\d{2,3}[.\-]\d{1,3}(?:[.\-]\d{1,3})?)\b")),
]


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extrae referencias normativas de la pregunta (sin LLM, sin costo).

    Devuelve [{"type", "value", "raw"}] deduplicado por (type, value). El tipo
    "ref" (número suelto "220.55") solo se emite si el valor no fue capturado ya
    por tabla/articulo/seccion — evita duplicar "tabla 220.55" + ref "220.55".
    El deep agent las usa como queries dirigidas (buscar la referencia literal
    rinde más que la pregunta completa) y quedan en la traza de Langfuse.
    """
    out: List[Dict[str, str]] = []
    seen: set = set()
    captured_values: set = set()
    for etype, rx in _ENTITY_PATTERNS:
        for m in rx.finditer(text or ""):
            value = m.group(1).strip()
            if etype == "norma":
                value = re.sub(r"\s+", " ", value.upper())
            elif etype in ("tabla", "articulo"):
                # "392.10 (A)" → "392.10(A)": misma normalización que el pipeline.
                value = re.sub(r"\s+", "", value)
            if etype == "ref" and value in captured_values:
                continue
            key = (etype, value.lower())
            if key in seen:
                continue
            seen.add(key)
            if etype in ("tabla", "articulo", "seccion"):
                captured_values.add(value)
            out.append({"type": etype, "value": value, "raw": m.group(0).strip()})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Resultado v3
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResultV3:
    """Contrato de salida del classifier_node (consumido por el deep agent)."""
    intent: str                # taxonomía _V3_INTENTS (+ "ambiguous" vía regex)
    source: str                # regex | llm | llm+escalated | *+guard
    route: str                 # smalltalk | ambiguous | out_of_domain | image_direct | retrieve
    wants_table: bool          # compat v1: rutea table_node / top_k tablas
    wants_full: bool           # compat v1: presupuesto amplio + salta enrich
    requires_rag: bool         # derivado por regla dura (route == "retrieve")
    response_format: str       # _V3_FORMATS
    output_length: str         # _V3_LENGTHS
    complexity: str            # _V3_COMPLEXITY ("high" = mensaje compuesto)
    needs_calculation: bool
    confidence: float          # 0.0–1.0 (regex determinístico → 1.0; fallback → 0.6)
    entities: List[Dict[str, str]] = field(default_factory=list)

    def to_state_meta(self) -> Dict[str, Any]:
        """Serializa los campos v3 para GraphState['intent_meta'] (JSON-safe)."""
        return {
            "requires_rag": self.requires_rag,
            "response_format": self.response_format,
            "output_length": self.output_length,
            "complexity": self.complexity,
            "needs_calculation": self.needs_calculation,
            "confidence": self.confidence,
            "entities": self.entities,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Clasificador LLM v3 (structured output + parser tolerante)
# ──────────────────────────────────────────────────────────────────────────────

_V3_JSON_SCHEMA = {
    "name": "intent_v3",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent", "response_format", "output_length",
            "complexity", "needs_calculation", "confidence",
        ],
        "properties": {
            "intent": {"type": "string", "enum": list(_V3_INTENTS)},
            "response_format": {"type": "string", "enum": list(_V3_FORMATS)},
            "output_length": {"type": "string", "enum": list(_V3_LENGTHS)},
            "complexity": {"type": "string", "enum": list(_V3_COMPLEXITY)},
            "needs_calculation": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
    },
}

_SYSTEM_PROMPT_V3 = (
    "Eres un clasificador de intención para un asistente de normativa eléctrica "
    "colombiana (RETIE y NTC 2050). Analiza la consulta y responde ÚNICAMENTE con un "
    "objeto JSON válido, sin markdown ni texto adicional, con esta forma exacta:\n"
    '{"intent": "...", "response_format": "...", "output_length": "...", '
    '"complexity": "...", "needs_calculation": false, "confidence": 0.0}\n\n'
    "intent — exactamente una de:\n"
    "- puntual: dato único y acotado (\"¿qué significa GFCI?\", \"define acometida\").\n"
    "- exhaustiva: contenido normativo COMPLETO de un tema o artículo (requisitos, "
    "exigencias, condiciones; \"qué exige/establece el artículo\"; \"todos los numerales\").\n"
    "- tabla: pide una tabla concreta o datos tabulares (\"la tabla 220.55\", "
    "\"tabla de calibres\", \"ampacidades en tabla\").\n"
    "- comparativa: contrastar dos o más normas, artículos, materiales o casos "
    "(\"diferencias entre RETIE y NTC 2050 en X\", \"cobre vs aluminio\").\n"
    "- procedimiento: pide pasos o trámite (\"¿cómo certifico una instalación?\", "
    "\"pasos para legalizar\").\n"
    "- verificacion: pregunta si algo cumple o es válido (\"¿puedo usar calibre 14 "
    "para tomas de 20 A?\", \"¿es obligatorio el GFCI en baños?\").\n"
    "- smalltalk: SOLO saludo/agradecimiento/despedida a secas, o pregunta sobre el "
    "propio bot (\"¿quién eres?\").\n"
    "- fuera_de_dominio: NO tiene NINGUNA relación con la electricidad, sus riesgos, "
    "sus efectos ni su normativa (películas, deportes, cocina, matemática general). "
    "OJO: el corpus del RETIE incluye también seguridad eléctrica y efectos de la "
    "corriente en el cuerpo humano (electropatología, fibrilación ventricular, "
    "tetanización), fenómenos eléctricos y atmosféricos (rayo, descargas, "
    "sobretensiones), protección contra rayos y las definiciones técnicas del Anexo "
    "General — preguntas sobre esos temas NUNCA son fuera_de_dominio aunque suenen "
    "a medicina o física.\n\n"
    "response_format — markdown | markdown_table | bullet_list | comparison | steps: "
    "cómo conviene presentar la respuesta (una pregunta puntual sobre datos tabulares "
    "puede llevar markdown_table).\n"
    "output_length — short | medium | detailed. 'detailed' si el usuario espera TODO "
    "el contenido de un tema sin omisiones.\n"
    "complexity — low | medium | high. 'high' si el mensaje contiene DOS O MÁS "
    "preguntas o peticiones distintas, o exige combinar varios artículos/normas.\n"
    "needs_calculation — true si la respuesta exige calcular con números del usuario "
    "(cargas, calibres por amperaje, distancias, factores de demanda aplicados).\n"
    "confidence — tu certeza en 'intent', de 0.0 a 1.0.\n\n"
    "REGLA CLAVE: si el mensaje MEZCLA un saludo o cortesía CON una pregunta o "
    "petición de información, clasifica por la PREGUNTA y NUNCA como smalltalk. "
    "En caso de duda entre fuera_de_dominio y otra categoría, elige la otra "
    "categoría y baja confidence.\n\n"
    "IMAGEN: si el mensaje contiene secciones como 'Usuario dijo sobre la imagen:', "
    "'Contenido interpretado de la imagen:' o 'Texto detectado en la imagen:', la "
    "consulta proviene de una imagen que envió el usuario: clasifica según la "
    "PREGUNTA del usuario sobre ese contenido (no según el tipo de objeto "
    "fotografiado). Una foto de un tablero, un conductor o un artículo normativo "
    "con una pregunta eléctrica NO es fuera_de_dominio.\n\n"
    "Ejemplos:\n"
    '- "hola" → {"intent":"smalltalk","response_format":"markdown","output_length":"short",'
    '"complexity":"low","needs_calculation":false,"confidence":0.99}\n'
    '- "hola, ¿qué es el RETIE?" → {"intent":"puntual","response_format":"markdown",'
    '"output_length":"short","complexity":"low","needs_calculation":false,"confidence":0.95}\n'
    '- "dame los requisitos de puesta a tierra" → {"intent":"exhaustiva",'
    '"response_format":"bullet_list","output_length":"detailed","complexity":"low",'
    '"needs_calculation":false,"confidence":0.9}\n'
    '- "pásame la tabla 220.55" → {"intent":"tabla","response_format":"markdown_table",'
    '"output_length":"detailed","complexity":"low","needs_calculation":false,"confidence":0.97}\n'
    '- "diferencias entre RETIE y NTC 2050 sobre GFCI" → {"intent":"comparativa",'
    '"response_format":"comparison","output_length":"detailed","complexity":"medium",'
    '"needs_calculation":false,"confidence":0.9}\n'
    '- "¿qué calibre necesito para una estufa de 12 kW a 240 V?" → {"intent":"verificacion",'
    '"response_format":"markdown","output_length":"medium","complexity":"medium",'
    '"needs_calculation":true,"confidence":0.85}\n'
    '- "recomiéndame una serie" → {"intent":"fuera_de_dominio","response_format":"markdown",'
    '"output_length":"short","complexity":"low","needs_calculation":false,"confidence":0.97}\n'
    '- "¿en qué consiste la fibrilación ventricular?" → {"intent":"puntual",'
    '"response_format":"markdown","output_length":"short","complexity":"low",'
    '"needs_calculation":false,"confidence":0.9}\n'
    '- "¿qué es un rayo?" → {"intent":"puntual","response_format":"markdown",'
    '"output_length":"short","complexity":"low","needs_calculation":false,"confidence":0.9}'
)


def _parse_intent_json(raw: str) -> Optional[Dict[str, Any]]:
    """Valida la salida del classifier v3. Tolerante a fences/texto alrededor.

    Devuelve el dict normalizado o None (→ el llamador cae al regex). `intent`
    inválido invalida todo; el resto de campos degrada a defaults seguros.
    Nunca lanza.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    intent = data.get("intent")
    if not isinstance(intent, str) or intent.strip().lower() not in _V3_INTENTS:
        return None

    def _pick(key: str, valid: tuple, default: str) -> str:
        v = data.get(key)
        return v if isinstance(v, str) and v in valid else default

    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.5
    conf = min(max(conf, 0.0), 1.0)

    return {
        "intent": intent.strip().lower(),
        "response_format": _pick("response_format", _V3_FORMATS, "markdown"),
        "output_length": _pick("output_length", _V3_LENGTHS, "medium"),
        "complexity": _pick("complexity", _V3_COMPLEXITY, "low"),
        "needs_calculation": data.get("needs_calculation") is True,
        "confidence": conf,
    }


def _llm_classify_v3(question: str, model: str) -> Optional[Dict[str, Any]]:
    """Una llamada del classifier v3. Devuelve el dict validado o None. Nunca lanza.

    Usa structured output nativo (json_schema estricto); si el modelo no lo
    soporta, create_chat_completion omite el parámetro y reintenta, y el parser
    tolerante cubre la salida sin esquema. La llamada queda registrada en
    Langfuse (modelo, usage) — v1 no la registraba y su costo real era invisible.
    """
    if not as_bool(getattr(settings, "INTENT_LLM_ENABLED", "true")):
        return None
    if not getattr(settings, "OPENAI_API_KEY", None):
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        resp = create_chat_completion(
            client,
            model=model,
            temperature=0.0,
            max_tokens=int(getattr(settings, "INTENT_V3_MAX_TOKENS", 200)),
            response_format={"type": "json_schema", "json_schema": _V3_JSON_SCHEMA},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_V3},
                {"role": "user", "content": (question or "").strip()[:500]},
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = _parse_intent_json(raw)
        try:
            from retie_agent.observability.obs import log_generation
            usage = getattr(resp, "usage", None)
            log_generation(
                None,
                name="intent.classify_v3",
                input_text=(question or "")[:500],
                output_text=raw,
                model=model,
                usage={
                    "input": getattr(usage, "prompt_tokens", None),
                    "output": getattr(usage, "completion_tokens", None),
                    "total": getattr(usage, "total_tokens", None),
                },
                metadata={"parsed": data is not None},
            )
        except Exception:
            pass
        return data
    except Exception as exc:
        logger.warning("intent v3 LLM classifier failed (%s): %s", model, exc)
        return None


def _regex_intent_v3(q: str) -> str:
    """Fallback determinístico v3. Prioridad:
    smalltalk → tabla → comparativa → exhaustiva → puntual.

    procedimiento/verificacion/fuera_de_dominio NO tienen vía regex: solo el LLM
    puede emitirlas (fallback conservador — degradan al léxico de negocio v1).
    """
    if _SMALLTALK_RE.match(_collapse_elongations(q)):
        return "smalltalk"
    if _TABLE_RE.search(q):
        return "tabla"
    if _COMPARE_RE.search(q):
        return "comparativa"
    if _FULL_RE.search(q):
        return "exhaustiva"
    return "puntual"


def _v3_defaults(intent: str, confidence: float) -> Dict[str, Any]:
    """Campos v3 por defecto cuando clasificó el regex (sin LLM)."""
    fmt = {
        "tabla": "markdown_table",
        "comparativa": "comparison",
        "exhaustiva": "bullet_list",
        "procedimiento": "steps",
    }.get(intent, "markdown")
    length = "detailed" if intent in ("tabla", "comparativa", "exhaustiva") else "medium"
    return {
        "intent": intent,
        "response_format": fmt,
        "output_length": length,
        "complexity": "low",
        "needs_calculation": False,
        "confidence": confidence,
    }


def _fixed_v3(intent: str, *, route: str, question: str) -> IntentResultV3:
    """Resultado determinístico de los fast-path regex (smalltalk/ambiguous)."""
    return IntentResultV3(
        intent=intent,
        source="regex",
        route=route,
        wants_table=False,
        wants_full=False,
        requires_rag=False,
        response_format="markdown",
        output_length="short",
        complexity="low",
        needs_calculation=False,
        confidence=1.0,
        entities=extract_entities(question),
    )


def classify_intent_v3(
    question: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    is_media: bool = False,
    channel: str = "telegram",
    media_source: Optional[str] = None,
) -> IntentResultV3:
    """Clasificación v3 con cascada de modelos y guardarraíles heredados de v1.

    Orden de decisión:
      1. Saludo confirmado por regex anclado → smalltalk (única puerta al saludo).
      2. Fragmento vago sin historial → ambiguous (se pide aclaración).
      3. LLM primario (INTENT_V3_MODEL → INTENT_MODEL). Si la confianza es baja
         (< INTENT_V3_ESCALATION_CONF) o el intent es sensible con confianza
         < INTENT_V3_SENSITIVE_CONF, re-clasifica con INTENT_V3_ESCALATION_MODEL.
      4. LLM caído/JSON inválido → regex v3 (léxico de negocio + comparativa).
      5. Guardarraíles: smalltalk del LLM sin confirmación del regex → falso
         positivo; fuera_de_dominio solo corta con vía LLM y confianza
         ≥ INTENT_OOD_MIN_CONF (si no, degrada a puntual + retrieve); si la
         consulta proviene de una IMAGEN (`media_source == "image"`),
         fuera_de_dominio nunca corta: rutea a "image_direct" para responder
         desde el contenido leído de la imagen (OCR/Vision), que viaja en la
         propia pregunta.

    `channel` queda reservado (hint de formato por canal); hoy no altera la
    clasificación. `media_source` es el origen del mensaje ("image" | "voice" |
    "text" | ...): solo "image" activa la ruta image_direct.
    """
    q = (question or "").strip()
    is_media = is_media or media_source in ("image", "voice")

    # 1. Saludo: solo el regex anclado puede emitirlo. No aplica a imagen/voz.
    #    Se normalizan alargamientos ("graciaaas" → "gracias") solo para el match.
    if not is_media and _SMALLTALK_RE.match(_collapse_elongations(q)):
        return _fixed_v3("smalltalk", route="smalltalk", question=q)

    # 2. Fragmento vago/incompleto sin historial que lo complete → aclarar.
    if not is_media and not history and _is_too_vague(q):
        return _fixed_v3("ambiguous", route="ambiguous", question=q)

    # 3. LLM primario + cascada (los checks de habilitación viven en
    #    _llm_classify_v3, como en v1: cualquier impedimento → None → regex).
    primary = (
        getattr(settings, "INTENT_V3_MODEL", None)
        or getattr(settings, "INTENT_MODEL", None)
        or getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    )
    data = _llm_classify_v3(q, primary)
    source = "llm"
    esc_model = getattr(settings, "INTENT_V3_ESCALATION_MODEL", None)
    if data is not None and esc_model and esc_model != primary:
        esc_conf = float(getattr(settings, "INTENT_V3_ESCALATION_CONF", 0.7))
        sens_conf = float(getattr(settings, "INTENT_V3_SENSITIVE_CONF", 0.85))
        dudoso = data["confidence"] < esc_conf or (
            data["intent"] in _V3_SENSITIVE_INTENTS and data["confidence"] < sens_conf
        )
        if dudoso:
            data2 = _llm_classify_v3(q, esc_model)
            if data2 is not None:
                data, source = data2, "llm+escalated"

    # 4. Fallback determinístico.
    if data is None:
        data = _v3_defaults(_regex_intent_v3(q), confidence=0.6)
        source = "regex"

    # 5a. Guardarraíl: smalltalk del LLM sin confirmación del regex (paso 1).
    if data["intent"] == "smalltalk":
        fb = _regex_intent_v3(q)
        data["intent"] = fb if fb != "smalltalk" else "puntual"
        source = f"{source}+guard"

    # 5b. Guardarraíl: fuera_de_dominio solo corta con vía LLM y confianza alta.
    # 5c. Guardarraíl imagen: si la consulta proviene de una imagen, lo leído por
    #     OCR/Vision viaja en la propia pregunta y suele contener la respuesta
    #     (p. ej. la clase de una etiqueta de eficiencia). El rechazo fijo la
    #     descartaría → se responde directo desde ese contenido, sin RAG.
    route = "retrieve"
    if data["intent"] == "fuera_de_dominio":
        if media_source == "image":
            route = "image_direct"
            source = f"{source}+guard"
        else:
            ood_conf = float(getattr(settings, "INTENT_OOD_MIN_CONF", 0.8))
            if source.startswith("llm") and data["confidence"] >= ood_conf:
                route = "out_of_domain"
            else:
                data["intent"] = "puntual"
                source = f"{source}+guard"

    # Derivaciones por regla dura (el LLM no puede contradecirlas).
    intent = data["intent"]
    wants_table = intent == "tabla" or data["response_format"] == "markdown_table"
    wants_full = intent in ("exhaustiva", "comparativa") or data["output_length"] == "detailed"
    if wants_full:
        data["output_length"] = "detailed"

    return IntentResultV3(
        intent=intent,
        source=source,
        route=route,
        wants_table=wants_table,
        wants_full=wants_full,
        requires_rag=(route == "retrieve"),
        response_format=data["response_format"],
        output_length=data["output_length"],
        complexity=data["complexity"],
        needs_calculation=data["needs_calculation"],
        confidence=data["confidence"],
        entities=extract_entities(q),
    )


