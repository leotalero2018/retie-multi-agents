from .pdfplumber_parser import PdfPlumberParser
from .pymupdf_parser import PyMuPDFParser
from .base import BaseParser

def get_parser(name: str) -> BaseParser:
    name = (name or "pdfplumber").lower()
    if name in ("plumber", "pdfplumber"):
        return PdfPlumberParser()
    if name in ("pymupdf", "fitz"):
        return PyMuPDFParser()
    raise ValueError(f"Parser no soportado: {name}")
