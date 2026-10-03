"""Readers for the documents financial research actually arrives in.

Annual reports and earnings releases are PDFs, data dumps are spreadsheets,
broker notes are Word files. These three functions turn each into text or a
DataFrame so pandas and NLP can pick them up from there.

Educational use only — not financial advice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def read_pdf(path: str | Path) -> str:
    """Extract the text of every page, joined with blank lines."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def read_excel(path: str | Path) -> pd.DataFrame:
    """Read the first sheet of an Excel workbook into a DataFrame."""
    return pd.read_excel(path)


def read_docx(path: str | Path) -> str:
    """Extract non-empty paragraphs of a Word document, joined with newlines."""
    import docx

    document = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs]
    return "\n".join(text for text in paragraphs if text)
