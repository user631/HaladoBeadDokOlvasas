from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
from fpdf import FPDF

FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def make_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(FONT_PATH))
    if title:
        pdf.set_title(title)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("DejaVu", size=12)
        if page_text:
            pdf.multi_cell(0, 8, page_text)
    pdf.output(str(path))
    return path


@pytest.fixture
def pdf_factory():
    return make_pdf


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    (tmp_path / "level.txt").write_text("Tisztelt Ügyfelünk!\nA számla mellékelve.", encoding="utf-8")
    (tmp_path / "adatok.csv").write_text("nev,kor\nAnna,31\nBéla,45\n", encoding="utf-8")
    make_pdf(tmp_path / "jelentes.pdf", ["Éves jelentés", "Második oldal"], title="Jelentés")
    (tmp_path / "kep.png").write_bytes(b"\x89PNG\r\n")
    nested = tmp_path / "almappa"
    nested.mkdir()
    (nested / "jegyzet.txt").write_text("Belső jegyzet", encoding="utf-8")
    return tmp_path
