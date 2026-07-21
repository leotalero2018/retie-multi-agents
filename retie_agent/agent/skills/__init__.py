# retie_agent/agent/skills/__init__.py
"""Registry determinista de skills del deep agent (FINAL_IMPLEMENTATION_NODES §5.2).

Cada skill es un archivo .md de este directorio: frontmatter con la metadata y
el cuerpo con las DIRECTIVAS que se inyectan al mensaje inicial del agente.
Añadir una skill = añadir un .md (+ tests); editar sus directivas = editar
Markdown, sin tocar código Python.

Formato del frontmatter (clave: valor, entre líneas `---`):
    name            nombre de la skill (default: nombre del archivo)
    intents         intents que la activan (lista separada por comas)
    formats         response_format del classifier que la activan (lista por comas)
    response_format formato de entrega esperado de la respuesta
    max_tokens      presupuesto de síntesis
    post_node       nodo de post-proceso del grafo ("table_node") — opcional
    default         "true" → skill por defecto cuando nada matchea

La selección NO la hace el agente: la hace resolve_skill() a partir del
IntentV3 del classifier_node, con degradación
(intent, fmt) → (intent, "*") → ("*", fmt) → DEFAULT_SKILL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class SkillSpec:
    name: str
    directives: str            # cuerpo del .md — bloque para el mensaje inicial
    response_format: str       # formato de entrega esperado
    max_tokens: int            # presupuesto de síntesis
    post_node: Optional[str] = None  # nodo de post-proceso ("table_node") o None


# Fallback de emergencia: si qa.md faltara o no parseara, el agente jamás se
# queda sin skill (mismo contenido que qa.md).
_EMERGENCY_DEFAULT = SkillSpec(
    name="qa",
    directives="Responde de forma directa y precisa la pregunta con base en la evidencia.",
    response_format="markdown",
    max_tokens=1600,
)


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Separa frontmatter (dict clave→valor) y cuerpo. Sin dependencias YAML."""
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    meta: Dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip()
    return meta, parts[2].strip()


def _csv(value: Optional[str]) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _load() -> Tuple[Dict[Tuple[str, str], SkillSpec], SkillSpec]:
    """Carga todos los .md del directorio y construye el registry.

    Una skill malformada se ignora con warning (nunca rompe el arranque): el
    grafo sigue funcionando con las demás y el default.
    """
    registry: Dict[Tuple[str, str], SkillSpec] = {}
    default: Optional[SkillSpec] = None
    for path in sorted(_SKILLS_DIR.glob("*.md")):
        try:
            meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
            if not body:
                logger.warning("skill %s sin directivas — ignorada", path.name)
                continue
            spec = SkillSpec(
                name=meta.get("name", path.stem),
                directives=body,
                response_format=meta.get("response_format", "markdown"),
                max_tokens=int(meta.get("max_tokens", 1600)),
                post_node=meta.get("post_node") or None,
            )
            for intent in _csv(meta.get("intents")):
                registry[(intent, "*")] = spec
            for fmt in _csv(meta.get("formats")):
                registry[("*", fmt)] = spec
            if meta.get("default", "").lower() in ("true", "1", "yes"):
                default = spec
        except Exception as exc:  # noqa: BLE001 — una skill rota no tumba el resto
            logger.warning("skill %s inválida (%s) — ignorada", path.name, exc)
    if default is None:
        logger.warning("ninguna skill marcada default — usando el fallback embebido")
    return registry, default or _EMERGENCY_DEFAULT


REGISTRY, DEFAULT_SKILL = _load()


def resolve_skill(intent: Optional[str], response_format: Optional[str]) -> SkillSpec:
    """Lookup con degradación: (intent, fmt) → (intent, "*") → ("*", fmt) → DEFAULT."""
    it = intent or "*"
    fmt = response_format or "*"
    for key in ((it, fmt), (it, "*"), ("*", fmt)):
        spec = REGISTRY.get(key)
        if spec is not None:
            return spec
    return DEFAULT_SKILL
