# app/utils/text.py
from __future__ import annotations

import re
from typing import List

# Try to use tiktoken if available; fall back gracefully otherwise
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _enc = None


def clean_text(text: str) -> str:
    """
    Normalize whitespace while keeping paragraph breaks modestly intact.
    (Your original collapsed all newlines; we now keep single \n between paragraphs.)
    """
    # Normalize CRLF -> LF
    text = re.sub(r"\r\n?", "\n", text or "")
    # Collapse tabs to spaces
    text = text.replace("\t", " ")
    # Collapse >2 blank lines to exactly one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces
    text = re.sub(r"[ \f\v]+", " ", text)
    # Trim spaces around newlines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    if _enc:
        return len(_enc.encode(text or ""))
    # Fallback heuristic ≈ 4 chars/token
    t = text or ""
    return max(1, (len(t) + 3) // 4)


def split_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split text by token count with overlap. If tiktoken is missing,
    we approximate by characters using 4 chars ≈ 1 token.
    """
    if not text:
        return []

    if _enc:
        tokens = _enc.encode(text)
        n = len(tokens)
        chunks: List[str] = []
        start = 0
        while start < n:
            end = min(start + chunk_tokens, n)
            chunk = _enc.decode(tokens[start:end])
            chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - overlap_tokens)
        return chunks

    # Fallback: character-based splitting (4 chars ≈ 1 token)
    approx_chars = max(1, chunk_tokens * 4)
    approx_overlap = max(0, overlap_tokens * 4)

    chunks: List[str] = []
    s = text
    i = 0
    while i < len(s):
        j = min(i + approx_chars, len(s))
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = max(0, j - approx_overlap)
    return chunks
