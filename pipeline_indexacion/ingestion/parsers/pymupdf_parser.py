from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from retie_agent.utils.text import clean_text
from .base import BaseParser

class PyMuPDFParser(BaseParser):
    def extract_pages(self, path: Path) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        with fitz.open(str(path)) as doc:
            for i, page in enumerate(doc, start=1):
                txt = page.get_text("text") or ""
                txt = clean_text(txt)
                if txt:
                    out.append((i, txt))
        return out
