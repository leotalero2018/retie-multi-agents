from dataclasses import dataclass
from typing import List
from app.utils.text import split_by_tokens
from app.config import settings


@dataclass
class Chunk:
    text: str
    chunk_id: int
    page: int | None = None


def make_chunks(text: str, page: int | None = None) -> List[Chunk]:
    chunks = split_by_tokens(
        text,
        chunk_tokens=settings.CHUNK_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP,
    )
    return [Chunk(t, i, page=page) for i, t in enumerate(chunks)]