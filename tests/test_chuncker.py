from app.utils.text import split_by_tokens


def test_split_by_tokens_basic():
    text = "Hola " * 100
    chunks = split_by_tokens(text, chunk_tokens=20, overlap_tokens=5)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], str)