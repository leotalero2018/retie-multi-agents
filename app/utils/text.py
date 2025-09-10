import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def clean_text(text: str) -> str:
    # Normaliza espacios y líneas
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def split_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int):
    """Divide texto por tokens con solapamiento.
    Devuelve lista de strings (chunks).
    """
    tokens = _enc.encode(text)
    n = len(tokens)
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_tokens, n)
        chunk = _enc.decode(tokens[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks