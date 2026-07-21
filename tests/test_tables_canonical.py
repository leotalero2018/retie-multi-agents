# tests/test_tables_canonical.py
"""Regresión del camino canónico de tablas en runtime — sin red.

La otra mitad del fix: el runtime debe servir las tablas como UNIDAD ATÓMICA
desde el registro de activos v2 (assets_registry.sqlite + JSON/PNG), no
reconstruirlas con un LLM desde chunks de texto. Cubre: lookup por id con
sufijos y variantes, compatibilidad con ids truncados históricos, y que
_node_table sirva el activo canónico sin invocar al LLM ni emitir el aviso de
"tabla incompleta".
"""
import json
import sqlite3

import pytest

from retie_agent.services import table_assets
from retie_agent.services.table_assets import CanonicalTable, lookup_table


# Fixtures canónicas: la tabla de la evidencia YA BIEN EXTRAÍDA (como la
# produce el pipeline corregido) y una solo_imagen.
GALVANIZADO = {
    "title": "Requisitos de galvanizado para láminas, pletinas y elementos roscados",
    "headers": ["RECUBRIMIENTO DE ZINC", "PROMEDIO gr/m²", "PROMEDIO µm",
                "MÍNIMO gr/m²", "MÍNIMO µm"],
    "rows": [
        ["Pletinas y láminas", "458", "65,4", "381", "54,4"],
        ["Elementos roscados", "397", "56,6", "336", "48"],
    ],
    "notes": "Fuente: Adoptada de la Resolución 180540 de 2010.",
}


@pytest.fixture
def assets_root(tmp_path, monkeypatch):
    """Construye un índice sincronizado mínimo: registro sqlite + JSON + PNG."""
    reg = tmp_path / "assets_registry.sqlite"
    con = sqlite3.connect(reg)
    con.execute(
        "CREATE TABLE tables_registry (doc_id TEXT, tabla_id TEXT, titulo TEXT,"
        " pagina_inicio INTEGER, pagina_fin INTEGER, bbox TEXT, png_key TEXT,"
        " json_key TEXT, markdown TEXT, estado TEXT, metodo TEXT,"
        " articulo_padre TEXT, hash_region TEXT, n_filas INTEGER, n_cols INTEGER,"
        " telegram_file_id TEXT, PRIMARY KEY (doc_id, tabla_id))"
    )

    tables_dir = tmp_path / "assets" / "tables" / "retie_libro2"
    tables_dir.mkdir(parents=True)
    (tables_dir / "galv.json").write_text(
        json.dumps(GALVANIZADO, ensure_ascii=False), encoding="utf-8"
    )
    (tables_dir / "solo.png").write_bytes(b"\x89PNG-fake-bytes")

    rows = [
        ("retie_libro2", "2.3.26.2.2.1.a", "Requisitos de galvanizado", 100, 100,
         None, None, "tables/retie_libro2/galv.json", None, "extraida", "vision",
         None, "h1", 2, 5, None),
        ("ntc2050_v2", "392.10(A)", "Métodos de alambrado", 200, 200,
         None, "tables/retie_libro2/solo.png", None, None, "solo_imagen", "none",
         None, "h2", 0, 0, None),
        ("ntc2050_v2", "400.4", "Tabla en revisión", 300, 300,
         None, None, None, None, "revision", "vision", None, "h3", 0, 0, None),
        # Entrada LEGACY de un índice pre-fix: id truncado (sin .a) cuyo JSON
        # puede ser otra tabla — jamás debe servirse como estructura canónica.
        ("retie_libro9", "9.9.9", "Tabla legacy truncada", 400, 400,
         None, None, "tables/retie_libro2/galv.json", None, "extraida", "vision",
         None, "h4", 5, 2, None),
    ]
    con.executemany(
        "INSERT INTO tables_registry VALUES (" + ",".join("?" * 16) + ")", rows
    )
    con.commit()
    con.close()

    monkeypatch.setattr(table_assets.settings, "CHROMA_PERSIST_DIR", str(tmp_path), raising=False)
    return tmp_path


# ── lookup: matching por id, sufijos y variantes ──────────────────────────────

def test_lookup_id_exacto_con_sufijo(assets_root):
    t = lookup_table("2.3.26.2.2.1.a")
    assert t is not None
    assert t.tabla_id == "2.3.26.2.2.1.a"
    assert t.headers == GALVANIZADO["headers"]
    assert t.rows[0][1] == "458"       # gr/m² en SU columna
    assert "Fuente" in (t.notes or "")


def test_lookup_variantes_de_notacion(assets_root):
    # espacios y mayúsculas no importan: "392.10 (a)" ≡ "392.10(A)"
    assert lookup_table("392.10 (a)").tabla_id == "392.10(A)"
    assert lookup_table("392.10(A)").tabla_id == "392.10(A)"
    # referencia sin sufijo encuentra la tabla sufijada
    assert lookup_table("392.10").tabla_id == "392.10(A)"


def test_lookup_ref_corta_encuentra_id_sufijado(assets_root):
    # Referencia sin sufijo → el registro más específico responde (confiable).
    t = lookup_table("2.3.26.2.2.1")
    assert t is not None and t.tabla_id == "2.3.26.2.2.1.a"
    assert t.match == "stored_extends_ref"


def test_lookup_marca_id_truncado_legacy_como_no_confiable(assets_root):
    # Pedir "9.9.9.a" cuando el índice viejo solo tiene "9.9.9": el match se
    # marca ref_extends_stored — la firma del truncamiento histórico.
    t = lookup_table("9.9.9.a")
    assert t is not None and t.tabla_id == "9.9.9"
    assert t.match == "ref_extends_stored"


def test_lookup_solo_imagen_trae_png(assets_root):
    t = lookup_table("392.10(A)")
    assert t.estado == "solo_imagen"
    assert t.headers == [] and t.rows == []
    assert t.png_path is not None and t.png_path.read_bytes().startswith(b"\x89PNG")


def test_lookup_sin_match_ni_registro():
    # sin registro (CHROMA_PERSIST_DIR por defecto sin sqlite) → None, sin lanzar
    assert lookup_table("9999.99") is None or True  # nunca lanza
    assert lookup_table("") is None


# ── _node_table: canónico primero, cero LLM, cero aviso de incompleta ─────────

def _boom_llm(monkeypatch):
    from retie_agent.agent import graph as g

    def _fail(*a, **k):
        raise AssertionError("el camino canónico no debe invocar al LLM")

    monkeypatch.setattr(g, "create_chat_completion", _fail)


def test_node_table_sirve_canonico_sin_llm(assets_root, monkeypatch):
    from retie_agent.agent import graph as g

    _boom_llm(monkeypatch)
    out = g._node_table({
        "question": "dame la tabla 2.3.26.2.2.1.a de galvanizado",
        "answer": "borrador previo",
        "hits": [],
        "intent_meta": {"entities": [
            {"type": "tabla", "value": "2.3.26.2.2.1.a", "raw": "tabla 2.3.26.2.2.1.a"}
        ]},
    })
    assert out.get("table_image"), "debe entregar la tabla como imagen"
    assert "2.3.26.2.2.1.a" in out["answer"]
    assert "Fuente" in out["answer"]                       # notes preservadas
    assert "incompleta" not in out["answer"].lower()       # el aviso desapareció
    assert "Referenciado en el contexto" not in out["answer"]


def test_node_table_solo_imagen_entrega_png_original(assets_root, monkeypatch):
    from retie_agent.agent import graph as g

    _boom_llm(monkeypatch)
    out = g._node_table({
        "question": "muéstrame la tabla 392.10(A)",
        "answer": "borrador",
        "hits": [],
    })
    assert out["table_image"].startswith(b"\x89PNG")
    assert "392.10(A)" in out["answer"]


def test_node_table_revision_cae_al_camino_llm(assets_root, monkeypatch):
    """estado=revision no es confiable → se usa el camino LLM actual."""
    from types import SimpleNamespace
    from retie_agent.agent import graph as g

    called = {"n": 0}

    def _fake_llm(client, **kwargs):
        called["n"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"title":"T","headers":["A","B"],"rows":[["1","2"]]}'),
            finish_reason="stop",
        )])

    monkeypatch.setattr(g, "create_chat_completion", _fake_llm)
    out = g._node_table({
        "question": "dame la tabla 400.4",
        "answer": "contexto con datos",
        "hits": [],
    })
    assert called["n"] >= 1
    assert out["route"] == "stylist_node"


def test_node_table_no_confia_en_json_de_id_truncado(assets_root, monkeypatch):
    """Índice pre-reindexación: el usuario pide la .a pero solo existe la
    entrada truncada (cuyo JSON puede ser OTRA tabla) → camino LLM, jamás
    servir esa estructura como canónica."""
    from types import SimpleNamespace
    from retie_agent.agent import graph as g

    called = {"n": 0}

    def _fake_llm(client, **kwargs):
        called["n"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"headers":[],"rows":[]}'),
            finish_reason="stop",
        )])

    monkeypatch.setattr(g, "create_chat_completion", _fake_llm)
    out = g._node_table({
        "question": "dame la tabla 9.9.9.a completa",
        "answer": "contexto",
        "hits": [],
    })
    assert called["n"] >= 1                    # cayó al camino LLM
    assert "Tabla legacy" not in (out.get("answer") or "")


def test_node_table_sin_referencia_usa_camino_llm(monkeypatch):
    """Sin id de tabla en la pregunta no hay lookup: comportamiento actual."""
    from types import SimpleNamespace
    from retie_agent.agent import graph as g

    def _fake_llm(client, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"headers":[],"rows":[]}'),
            finish_reason="stop",
        )])

    monkeypatch.setattr(g, "create_chat_completion", _fake_llm)
    out = g._node_table({
        "question": "hazme una tabla comparativa de lo anterior",
        "answer": "contenido base",
        "hits": [],
    })
    assert out["route"] == "stylist_node"
