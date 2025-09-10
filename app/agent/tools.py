# Herramientas de ejemplo (stubs). Conéctalas a tus fuentes/datos.
from typing import Any, Dict


def get_full_page(source: str, page: int) -> Dict[str, Any]:
    # TODO: recuperar la página completa para dar contexto ampliado
    return {"source": source, "page": page, "content": "<contenido completo no implementado>"}


def compare_versions(query: str) -> Dict[str, Any]:
    # TODO: ejemplo para comparar versiones de norma
    return {"result": f"Comparación de versiones para: {query} (stub)"}


def search_by_date(topic: str, year: int) -> Dict[str, Any]:
    # TODO: ejemplo de búsqueda por fecha
    return {"result": f"Búsqueda por fecha {year} sobre {topic} (stub)"}