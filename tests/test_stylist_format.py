from retie_agent.agent.graph import (
    _format_for_channel,
    _format_markdownish_as_telegram_html,
    _node_stylist,
)


def test_telegram_formatter_converts_markdown_to_safe_html():
    text = "# Título\n\n**Importante** y *detalle* con `RETIE` & <tag>"

    out = _format_markdownish_as_telegram_html(text)

    assert "<b>Título</b>" in out
    assert "<b>Importante</b>" in out
    assert "<i>detalle</i>" in out
    assert "<code>RETIE</code>" in out
    assert "&amp;" in out
    # <tag> no es una etiqueta Telegram soportada: se descarta en vez de
    # mostrarse escapada (&lt;tag&gt;) — ver test_raw_html_from_model_*.
    assert "<tag>" not in out
    assert "&lt;tag&gt;" not in out


def test_raw_html_from_model_preserves_safe_tags_and_drops_the_rest():
    """El modelo (o una fuente secundaria como Gemini/NotebookLM) a veces
    escribe HTML crudo en vez de **negrilla** markdown. Regresión del bug
    reportado en producción: html.escape() mostraba <b> como &lt;b&gt; visible
    en vez de negrilla real."""
    text = "El <b>Artículo 502</b> aplica en <div class=x>áreas Clase II</div>."

    out = _format_markdownish_as_telegram_html(text)

    assert "<b>Artículo 502</b>" in out
    assert "&lt;b&gt;" not in out
    assert "<div" not in out and "&lt;div" not in out
    assert "áreas Clase II" in out


def test_raw_html_does_not_swallow_numeric_comparisons():
    text = "Aplica si la tensión es <600V y **crítico**."

    out = _format_markdownish_as_telegram_html(text)

    assert "&lt;600V" in out
    assert "<b>crítico</b>" in out


def test_telegram_formatter_preserves_pre_blocks():
    text = "Antes\n\n<pre>A  B\n1  2</pre>\n\nDespués **ok**"

    out = _format_markdownish_as_telegram_html(text)

    assert "<pre>A  B\n1  2</pre>" in out
    assert "Después <b>ok</b>" in out


def test_format_for_channel_keeps_channels_separate():
    text = "**Negrilla** <b>html ajeno</b>"

    telegram, telegram_format = _format_for_channel(text, "telegram")
    whatsapp, whatsapp_format = _format_for_channel(text, "whatsapp")

    assert telegram_format == "telegram_html"
    assert "<b>Negrilla</b>" in telegram
    # <b> crudo del modelo es una etiqueta Telegram soportada: se preserva
    # como negrilla real, no se escapa a &lt;b&gt; visible.
    assert "<b>html ajeno</b>" in telegram
    assert "&lt;b&gt;" not in telegram
    assert whatsapp_format == "whatsapp_text"
    assert "<b>" not in whatsapp


def test_stylist_payload_declares_channel_format_and_formats_sources():
    out = _node_stylist(
        {
            "answer": "## Respuesta\n\n**Sí**, aplica.",
            "question": "¿Aplica?",
            "metadata": {"channel": "telegram"},
            "hits": [
                {
                    "text": "Artículo 1.2.3 Requisito de prueba",
                    "meta": {"source": "RETIE <2024>.pdf", "page": 10},
                }
            ],
        }
    )

    payload = out["answer"]

    assert payload["channel"] == "telegram"
    assert payload["format"] == "telegram_html"
    assert "<b>Respuesta</b>" in payload["formatted_response"]
    assert "<b>Sí</b>" in payload["formatted_response"]
    assert "RETIE &lt;2024&gt;" in payload["sources_text"]
