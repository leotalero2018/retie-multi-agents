from dataclasses import dataclass
from typing import List
from retie_agent.utils.text import split_by_tokens
from retie_agent.config import settings


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