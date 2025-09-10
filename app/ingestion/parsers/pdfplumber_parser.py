from pathlib import Path
from typing import List, Tuple
import pdfplumber
from app.utils.text import clean_text
from .base import BaseParser

class PdfPlumberParser(BaseParser):
    def extract_pages(self, path: Path) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                txt = clean_text(txt)
                if txt:
                    out.append((i, txt))
        return out
