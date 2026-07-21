"""Helper de GPT-4o Vision para el pipeline (extracción de tablas y captions).

Presupuesto controlado por corrida (V2_VISION_MAX_CALLS) y resultados
cacheados vía hash_region en el registro — cada región se paga UNA vez.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_TABLE_PROMPT = (
    "La imagen contiene una página o recorte de un documento normativo eléctrico en español. "
    "Extrae LA TABLA principal (la que corresponde al título 'Tabla {tabla_id}' si aparece).\n"
    "Responde ÚNICAMENTE con JSON válido, sin markdown:\n"
    '{{"title": "...", "headers": ["..."], "rows": [["..."]], "notes": "..."}}\n'
    "Reglas ESTRICTAS:\n"
    "- Reproduce TODAS las filas y TODOS los valores EXACTAMENTE como aparecen "
    "(números, unidades, rangos como '26-30', fórmulas, notas al pie referenciadas).\n"
    "- NO resumas, NO omitas filas, NO corrijas ortografía.\n"
    "- HEADERS DE VARIOS NIVELES: si una celda padre abarca subcolumnas "
    "(p. ej. 'PROMEDIO' sobre 'gr/m²' y 'µm'), combina padre y subcolumna en UN "
    "header por columna final: 'PROMEDIO gr/m²', 'PROMEDIO µm'. El número de "
    "elementos de 'headers' debe ser EXACTAMENTE el número de celdas de cada fila "
    "de datos — cada valor debe quedar bajo SU columna.\n"
    "- Celdas combinadas verticales: repite el valor en cada fila que abarca.\n"
    "- CELDAS VACÍAS LEGÍTIMAS: si la celda está vacía o tiene un guion (—), "
    "escribe '—'. NUNCA la rellenes, no inventes contenido ni celdas de relleno.\n"
    "- Notas al pie y la línea 'Fuente: ...': van COMPLETAS en 'notes', no como filas.\n"
    "- Si la tabla continúa fuera de la imagen, extrae lo visible (notes: 'continúa').\n"
    "- Si NO hay tabla legible: {{\"headers\": [], \"rows\": []}}."
)

_CAPTION_PROMPT = (
    "La imagen es una figura de un documento normativo eléctrico colombiano (RETIE/NTC 2050). "
    "Escribe en español una descripción técnica de 1-3 frases: qué muestra, qué elementos "
    "incluye y para qué sirve. Si la figura tiene número/título visible, inclúyelo al inicio. "
    "Responde SOLO con la descripción."
)


class VisionBudget:
    """Contador compartido de llamadas Vision por corrida."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True


def _client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from retie_agent.config import settings
            api_key = settings.OPENAI_API_KEY
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no configurada — Vision no disponible")
    return OpenAI(api_key=api_key)


def _data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _close_truncated_json(s: str) -> str:
    """Repara JSON cortado por el límite de tokens: balancea llaves/corchetes,
    cierra strings abiertos y descarta una coma/fila final incompleta."""
    in_str = False
    esc = False
    stack: List[str] = []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
    repaired = s
    if in_str:
        repaired += '"'
    repaired = repaired.rstrip()
    while repaired and repaired[-1] == ",":
        repaired = repaired[:-1].rstrip()
    for opener in reversed(stack):
        repaired += "]" if opener == "[" else "}"
    return repaired


def _parse_json_lenient(raw: str) -> Optional[dict]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:]
    end = candidate.rfind("}")
    snippet = candidate[: end + 1] if end != -1 else candidate
    for attempt in (snippet, _close_truncated_json(candidate)):
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if attempt is not snippet:
                # JSON reparado: la última fila pudo perderse → marcar para revision
                data["_truncated"] = True
            return data
    return None


def extract_table_vision(png_bytes: bytes, tabla_id: str, model: str,
                         budget: VisionBudget) -> Optional[Dict[str, Any]]:
    """Devuelve {title, headers, rows, notes} o None (sin presupuesto / fallo)."""
    if not budget.take():
        log.warning("[VISION] presupuesto agotado (%d llamadas) — tabla %s queda solo_imagen",
                    budget.max_calls, tabla_id)
        return None
    try:
        resp = _client().chat.completions.create(
            model=model, temperature=0.0, max_tokens=4000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _TABLE_PROMPT.format(tabla_id=tabla_id)},
                    {"type": "image_url", "image_url": {"url": _data_url(png_bytes), "detail": "high"}},
                ],
            }],
        )
        raw = resp.choices[0].message.content or ""
        truncated = getattr(resp.choices[0], "finish_reason", "") == "length"
        data = _parse_json_lenient(raw)
        if not data:
            log.warning("[VISION] tabla %s: JSON no parseable (%d chars)", tabla_id, len(raw))
            return None
        headers = data.get("headers") or []
        rows = data.get("rows") or []
        if not headers or not rows:
            return None
        if truncated:
            data["notes"] = ((data.get("notes") or "") + " [extracción truncada]").strip()
            data["_truncated"] = True
        return data
    except Exception as exc:
        log.warning("[VISION] tabla %s falló: %s", tabla_id, exc)
        return None


def caption_figure_vision(png_bytes: bytes, model: str, budget: VisionBudget) -> Optional[str]:
    if not budget.take():
        return None
    try:
        resp = _client().chat.completions.create(
            model=model, temperature=0.0, max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": _data_url(png_bytes), "detail": "low"}},
                ],
            }],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        log.warning("[VISION] caption falló: %s", exc)
        return None
