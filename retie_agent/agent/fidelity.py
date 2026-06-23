# app/agent/fidelity.py
"""Verificador de fidelidad para el enriquecimiento (TICKET-003 / H-903).

El nodo de enriquecimiento reescribe la respuesta con un Assistant. En un sistema
normativo eso es peligroso: parafrasear puede **omitir literales**, **alterar
valores numéricos** o **introducir contenido externo** a la evidencia recuperada.

Este módulo extrae los TOKENS CRÍTICOS DE FIDELIDAD de un texto:

  • valores numéricos (120, 0.45, 250.122, 220.55, 2024…) — cubre tensiones,
    corrientes, calibres, distancias, porcentajes, años, numerales y nº de tabla;
  • marcadores de lista / literales: `a)`, `b.`, `(C)`, `iv)`, `12)`…;

y permite verificar que la versión enriquecida los conserve EXACTAMENTE (mismo
multiconjunto). Si difieren —por omisión o por adición— el llamador descarta el
enriquecimiento y entrega el borrador original (siempre fiel a la evidencia).

El verificador es deliberadamente estricto y simétrico: un falso rechazo solo
cuesta el pulido estilístico (se sirve el borrador correcto), nunca la fidelidad.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict

# Valores numéricos: enteros y decimales con separadores internos `.` o `,`.
# Captura 120, 0.45, 250.122, 220.55, 3,5, 2.024 — valores, numerales, nº de tabla.
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

# Marcador de lista con paréntesis de cierre: `a)`, `f)`, `iv)`, `12)`.
# El lookbehind `(?<!\()` evita contar dos veces los literales entre paréntesis
# `(a)` (los captura _PAREN_RE).
_LIST_RE = re.compile(r"(?<!\()\b([A-Za-z]{1,3}|\d{1,2})\)")

# Literal entre paréntesis: `(C)`, `(A)`, `(1)` — referencias normativas inline.
_PAREN_RE = re.compile(r"\(([A-Za-z]{1,3}|\d{1,2})\)")

# Marcador de lista con punto al inicio de línea: `1.`, `a.` (seguido de espacio).
_DOT_RE = re.compile(r"(?m)^[ \t]*([A-Za-z]{1,3}|\d{1,2})\.(?=\s)")


def fidelity_tokens(text: str) -> Counter:
    """Multiconjunto de tokens críticos (números + marcadores de lista) del texto."""
    text = text or ""
    counts: Counter = Counter()
    for m in _NUM_RE.findall(text):
        counts[f"n:{m.replace(',', '.')}"] += 1
    for rx in (_LIST_RE, _PAREN_RE, _DOT_RE):
        for m in rx.findall(text):
            counts[f"l:{m.lower()}"] += 1
    return counts


def fidelity_diff(draft: str, enriched: str) -> Dict[str, Dict[str, int]]:
    """Reporta qué tokens se omitieron y cuáles se inventaron (para la traza)."""
    d = fidelity_tokens(draft)
    e = fidelity_tokens(enriched)
    return {
        "missing": dict(d - e),  # estaban en el borrador y desaparecieron (omisión)
        "added": dict(e - d),    # aparecieron en el enriquecido (posible alucinación)
    }


def enrichment_preserves_fidelity(draft: str, enriched: str) -> bool:
    """True si el enriquecido conserva EXACTAMENTE los tokens críticos del borrador.

    Multiconjunto igual ⇒ no se omitió ni se inventó ningún valor numérico,
    referencia normativa ni elemento de lista.
    """
    return fidelity_tokens(draft) == fidelity_tokens(enriched)
