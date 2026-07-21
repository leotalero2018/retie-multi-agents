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
    assert "&lt;tag&gt;" in out


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
    assert "&lt;b&gt;html ajeno&lt;/b&gt;" in telegram
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
