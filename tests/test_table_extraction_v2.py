# tests/test_table_extraction_v2.py
"""Regresión de extracción de tablas (pipeline v2) — sin red, sin PDFs.

Fixtures = los 4 formatos del criterio de aceptación, como rejillas crudas
tal cual las devuelve pdfplumber (filas de celdas, headers multinivel en filas
separadas, celdas vacías por spans):

  1. simple: 2 columnas, header plano (ITEM | VALOR)
  2. header de 2 niveles (Tabla 2.3.26.2.2.1.a galvanizado: PROMEDIO/MÍNIMO
     abarcan gr/m² y µm) — la evidencia del bug de columnas corridas
  3. header de 3 niveles con notación "1,5 – (30 Sd)" y footnotes
  4. celdas vacías legítimas (—) que NO deben rellenarse

Más la firma del segundo bug: prosa a dos columnas partida como tabla falsa
(Tabla 392.10(A)) que la estrategia "text" debe RECHAZAR, y el truncamiento de
IDs con sufijo en las anclas.
"""
import pytest

from pipeline_indexacion.v2.tables import (
    _ANCHOR_RE,
    _looks_like_prose,
    _normalize_rows,
    find_anchors,
    merge_multilevel_headers,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures: rejillas crudas estilo pdfplumber
# ══════════════════════════════════════════════════════════════════════════════

# 1. Simple — 2 columnas, header plano.
GRID_SIMPLE = [
    ["ITEM", "VALOR"],
    ["Presión del viento", "60 km/m²"],
    ["Carga de rotura", "150 kg"],
    ["Límite mínimo de fluencia del acero", "18,4 kg/mm² (180 MN/m²)"],
    ["Resistencia a la tracción", "34,7 kg/mm² (340 MN/m²)"],
    ["Elongación", "30% en 50 mm (2 pulgadas)"],
]

# 2. Dos niveles — Tabla 2.3.26.2.2.1.a (galvanizado). El padre combinado deja
# celdas vacías en su fila; las unidades viven en la fila siguiente.
GRID_GALVANIZADO = [
    ["RECUBRIMIENTO DE ZINC", "PROMEDIO", "", "MÍNIMO", ""],
    ["", "gr/m²", "µm", "gr/m²", "µm"],
    ["Pletinas y láminas", "458", "65,4", "381", "54,4"],
    ["Elementos roscados", "397", "56,6", "336", "48"],
]

GALVANIZADO_EXPECTED_HEADERS = [
    "RECUBRIMIENTO DE ZINC",
    "PROMEDIO gr/m²",
    "PROMEDIO µm",
    "MÍNIMO gr/m²",
    "MÍNIMO µm",
]
GALVANIZADO_EXPECTED_ROWS = [
    ["Pletinas y láminas", "458", "65,4", "381", "54,4"],
    ["Elementos roscados", "397", "56,6", "336", "48"],
]

# 3. Tres niveles + notación "1,5 – (30 Sd)" + celdas vacías legítimas (—).
GRID_TRES_NIVELES = [
    ["CLASE", "AISLAMIENTO", "", "", ""],
    ["", "INTERIOR", "", "EXTERIOR", ""],
    ["", "kV", "mm", "kV", "mm"],
    ["A", "1,5 – (30 Sd)", "—", "2,4", "25"],
    ["B", "3,6", "30", "—", "40"],
]

TRES_NIVELES_EXPECTED_HEADERS = [
    "CLASE",
    "AISLAMIENTO INTERIOR kV",
    "AISLAMIENTO INTERIOR mm",
    "AISLAMIENTO EXTERIOR kV",
    "AISLAMIENTO EXTERIOR mm",
]

# 5. Firma del bug de la 392.10(A): prosa a dos columnas cortada en celdas que
# parten palabras (extraído literalmente del JSON canónico corrupto real).
GRID_PROSA_392 = [
    ["se indica a cont", "inuación.", "", ""],
    ["(1) Conduct", "ores individua", "les. Debe perm", "itirse la ins-"],
    ["talación de cab", "les de un solo", "conductor, de", "acuerdo con"],
    ["lo establecido e", "n 392.10(B)(1)", "(a) hasta (B)(1", ")(c), como se"],
    ["describe a conti", "nuación.", "", ""],
    ["(a) Un cable", "de un solo con", "ductor debe se", "r de sección"],
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Simple: header plano pasa intacto
# ══════════════════════════════════════════════════════════════════════════════

def test_simple_header_plano_intacto():
    headers, rows = _normalize_rows(GRID_SIMPLE)
    assert headers == ["ITEM", "VALOR"]
    assert len(rows) == 5
    assert rows[0] == ["Presión del viento", "60 km/m²"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dos niveles: cero columnas perdidas, cada valor en SU columna
# ══════════════════════════════════════════════════════════════════════════════

def test_galvanizado_headers_compuestos():
    headers, rows = _normalize_rows(GRID_GALVANIZADO)
    assert headers == GALVANIZADO_EXPECTED_HEADERS


def test_galvanizado_valores_en_su_columna():
    """La evidencia del bug: gr/m² desaparecía y µm se corría a la izquierda."""
    headers, rows = _normalize_rows(GRID_GALVANIZADO)
    assert rows == GALVANIZADO_EXPECTED_ROWS
    # Verificación explícita celda a celda de la evidencia:
    fila = dict(zip(headers, rows[0]))
    assert fila["PROMEDIO gr/m²"] == "458"
    assert fila["PROMEDIO µm"] == "65,4"
    assert fila["MÍNIMO gr/m²"] == "381"
    assert fila["MÍNIMO µm"] == "54,4"
    fila2 = dict(zip(headers, rows[1]))
    assert fila2["PROMEDIO gr/m²"] == "397"
    assert fila2["MÍNIMO µm"] == "48"


def test_galvanizado_cero_filas_ni_columnas_faltantes():
    headers, rows = _normalize_rows(GRID_GALVANIZADO)
    assert len(headers) == 5          # las 5 columnas reales
    assert len(rows) == 2             # las 2 filas de datos reales
    assert all(len(r) == 5 for r in rows)
    assert all(all(c for c in r) for r in rows)  # ninguna celda vaciada


# ══════════════════════════════════════════════════════════════════════════════
# 3. Tres niveles + footnotes/notación
# ══════════════════════════════════════════════════════════════════════════════

def test_tres_niveles_headers_compuestos():
    headers, rows = _normalize_rows(GRID_TRES_NIVELES)
    assert headers == TRES_NIVELES_EXPECTED_HEADERS
    assert len(rows) == 2


def test_tres_niveles_notacion_preservada():
    headers, rows = _normalize_rows(GRID_TRES_NIVELES)
    fila = dict(zip(headers, rows[0]))
    assert fila["AISLAMIENTO INTERIOR kV"] == "1,5 – (30 Sd)"   # notación intacta


# ══════════════════════════════════════════════════════════════════════════════
# 4. Celdas vacías legítimas (—): se preservan, no se rellenan
# ══════════════════════════════════════════════════════════════════════════════

def test_celdas_vacias_legitimas_no_se_rellenan():
    headers, rows = _normalize_rows(GRID_TRES_NIVELES)
    fila_a = dict(zip(headers, rows[0]))
    fila_b = dict(zip(headers, rows[1]))
    assert fila_a["AISLAMIENTO INTERIOR mm"] == "—"
    assert fila_b["AISLAMIENTO EXTERIOR kV"] == "—"


def test_fila_de_datos_con_guiones_no_es_subheader():
    """Una fila de datos con celdas '—' no debe absorberse como nivel de header."""
    grid = [
        ["MATERIAL", "TENSIÓN", "CORRIENTE"],
        ["Cobre", "—", "25"],
        ["Aluminio", "13,2", "—"],
    ]
    headers, rows = _normalize_rows(grid)
    assert headers == ["MATERIAL", "TENSIÓN", "CORRIENTE"]
    assert len(rows) == 2
    assert rows[0] == ["Cobre", "—", "25"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Prosa partida como tabla falsa (392.10(A)) → RECHAZADA
# ══════════════════════════════════════════════════════════════════════════════

def test_prosa_392_se_detecta():
    headers, rows = GRID_PROSA_392[0], GRID_PROSA_392[1:]
    assert _looks_like_prose(headers, rows) is True


def test_tablas_reales_no_se_confunden_con_prosa():
    h1, r1 = _normalize_rows(GRID_GALVANIZADO)
    assert _looks_like_prose(h1, r1) is False
    h2, r2 = _normalize_rows(GRID_TRES_NIVELES)
    assert _looks_like_prose(h2, r2) is False
    h3, r3 = _normalize_rows(GRID_SIMPLE)
    assert _looks_like_prose(h3, r3) is False


# ══════════════════════════════════════════════════════════════════════════════
# 6. Anclas: IDs con sufijo capturados completos
# ══════════════════════════════════════════════════════════════════════════════

def test_ancla_retie_sufijo_punto_letra():
    m = _ANCHOR_RE.match("Tabla 2.3.26.2.2.1.a. Requisitos de galvanizado para láminas")
    assert m is not None
    assert m.group(1) == "2.3.26.2.2.1.a"
    assert "Requisitos" in m.group(2)


def test_ancla_ntc_sufijo_parentesis():
    m = _ANCHOR_RE.match("Tabla 392.10 (A) Métodos de alambrado")
    assert m is not None
    assert m.group(1).replace(" ", "") == "392.10(A)"
    m2 = _ANCHOR_RE.match("Tabla 392.10(A) Métodos de alambrado")
    assert m2 is not None and m2.group(1) == "392.10(A)"


def test_ancla_sin_sufijo_no_absorbe_titulo():
    """"Tabla 220.55. Factores…" NO debe capturar '.F' como sufijo."""
    m = _ANCHOR_RE.match("Tabla 220.55. Factores de demanda para estufas")
    assert m is not None
    assert m.group(1) == "220.55"


def test_find_anchors_ids_distintos_para_sufijos():
    """Las tablas .a y .b ya NO se fusionan bajo un mismo id (antes solo
    sobrevivía la primera y se extraía la tabla equivocada)."""
    pages = {
        "10": "Tabla 2.3.26.2.2.1.a. Requisitos de galvanizado\nTabla 2.3.26.2.2.1.b. Otras exigencias",
        "12": "Tabla 392.10 (A) Métodos de alambrado",
    }
    anchors = find_anchors(pages)
    ids = {a["tabla_id"] for a in anchors}
    assert ids == {"2.3.26.2.2.1.a", "2.3.26.2.2.1.b", "392.10(A)"}
    # El título ya no arrastra la letra del sufijo.
    galv = next(a for a in anchors if a["tabla_id"] == "2.3.26.2.2.1.a")
    assert galv["titulo"].startswith("Requisitos")


# ══════════════════════════════════════════════════════════════════════════════
# 7. merge directo (API pública del helper)
# ══════════════════════════════════════════════════════════════════════════════

def test_merge_sin_subheader_devuelve_header_plano():
    headers, body = merge_multilevel_headers(GRID_SIMPLE)
    assert headers == ["ITEM", "VALOR"]
    assert len(body) == 5


def test_merge_nunca_se_come_todas_las_filas():
    # Grid degenerado: header + una sola fila que PARECE subheader → debe
    # conservarse como datos (siempre queda al menos una fila de cuerpo).
    grid = [["A", "B", ""], ["", "x", "y"]]
    headers, body = merge_multilevel_headers(grid)
    assert body == [["", "x", "y"]]
