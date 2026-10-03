"""Sample documents for notebook 08, generated locally so runs stay offline.

Three synthetic SampleCo documents stand in for the real thing (annual report
PDFs, spreadsheet exports, Word research notes). Re-runnable: it overwrites
whatever is in ``sample_docs/`` and needs no arguments.

Usage: ``uv run python scripts/make_sample_docs.py``
"""

from __future__ import annotations

from pathlib import Path

import docx
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_docs"

HEADLINE = "SampleCo Q3 2025 Earnings Summary"
FINANCIALS = [
    ["Metric", "Q3 2025", "Q2 2025"],
    ["Revenue", "$1.20B", "$1.04B"],
    ["Net income", "$240M", "$205M"],
    ["EPS", "$1.20", "$1.02"],
]


def write_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = [
        Paragraph(HEADLINE, styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "SampleCo reported third-quarter revenue of $1.2 billion, up 15% "
            "year over year, driven by strong demand in its core subscription "
            "business.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Net income was $240 million, or $1.20 per diluted share. "
            "Management reiterated full-year guidance and highlighted margin "
            "expansion as input costs eased.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Table(FINANCIALS, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ])),
    ]
    SimpleDocTemplate(str(path), pagesize=LETTER, title=HEADLINE).build(story)


def write_xlsx(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Income Statement"
    sheet.append(["Line item ($M)", "Q1 2025", "Q2 2025", "Q3 2025"])
    sheet.append(["Revenue", 900, 1040, 1200])
    sheet.append(["Gross Profit", 540, 640, 750])
    sheet.append(["Net Income", 170, 205, 240])
    workbook.save(path)


def write_docx(path: Path) -> None:
    document = docx.Document()
    document.add_heading("SampleCo Equity Research Note", level=1)
    document.add_paragraph(
        "SampleCo delivered a clean quarter with revenue of $1.2B (+15% YoY) "
        "and net income of $240M. Gross margin held above 62%, consistent with "
        "the prior three quarters."
    )
    document.add_paragraph(
        "Risks: customer concentration in the top ten accounts and a strong "
        "dollar could pressure reported growth. We remain constructive on the "
        "subscription transition and see margin upside if input costs stay flat."
    )
    document.save(str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        "earnings_note.pdf": write_pdf,
        "quarterly_financials.xlsx": write_xlsx,
        "research_note.docx": write_docx,
    }
    for name, writer in targets.items():
        path = OUT_DIR / name
        writer(path)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
