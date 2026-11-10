from __future__ import annotations

from pathlib import Path

import pytest

from docproc import Document, iter_documents, load_document, supported_extensions
from docproc.config import SAMPLES_DIR
from docproc.exceptions import DocumentLoadError, EmptyDocumentError, UnsupportedFormatError
from docproc.loaders import (
    BaseLoader,
    CsvLoader,
    PdfLoader,
    TxtLoader,
    get_loader,
    iter_files,
    register_loader,
    unregister_loader,
)


class TestTxtLoader:
    def test_utf8(self, tmp_path: Path):
        path = tmp_path / "a.txt"
        path.write_text("Árvíztűrő tükörfúrógép", encoding="utf-8")
        document = load_document(path)
        assert document.text == "Árvíztűrő tükörfúrógép"
        assert document.metadata["encoding"] == "utf-8"
        assert document.file_type == "txt"

    def test_utf8_bom(self, tmp_path: Path):
        path = tmp_path / "bom.txt"
        path.write_bytes("﻿Szöveg BOM-mal".encode("utf-8"))
        document = load_document(path)
        assert document.text == "Szöveg BOM-mal"
        assert document.metadata["encoding"] == "utf-8-sig"

    def test_cp1250(self, tmp_path: Path):
        path = tmp_path / "regi.txt"
        path.write_bytes("Hosszú ő és ű betűk".encode("cp1250"))
        document = load_document(path)
        assert document.text == "Hosszú ő és ű betűk"
        assert document.metadata["encoding"] == "cp1250"

    def test_newlines_are_normalized(self, tmp_path: Path):
        path = tmp_path / "win.txt"
        path.write_bytes(b"egy\r\nketto\rharom")
        assert load_document(path).text == "egy\nketto\nharom"

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "ures.txt"
        path.write_text("   \n  ", encoding="utf-8")
        with pytest.raises(EmptyDocumentError):
            load_document(path)


class TestCsvLoader:
    def test_comma_separated(self, tmp_path: Path):
        path = tmp_path / "adatok.csv"
        path.write_text("nev,kor,varos\nAnna,31,Szeged\nBéla,45,Pécs\n", encoding="utf-8")
        document = load_document(path)
        assert document.metadata["delimiter"] == ","
        assert document.metadata["row_count"] == 2
        assert document.metadata["columns"] == ["nev", "kor", "varos"]
        assert document.metadata["numeric_columns"] == ["kor"]
        assert document.table is not None
        assert document.table["kor"].sum() == 76
        assert "Anna | 31 | Szeged" in document.text

    def test_hungarian_number_format(self, tmp_path: Path):
        path = tmp_path / "hu.csv"
        path.write_text(
            "Név;Összeg;Telefon\nKiss Anna;1 250 000,50;+36 30 123 4567\nNagy Béla;980,25;+36 1 234 5678\n",
            encoding="utf-8",
        )
        document = load_document(path)
        assert document.metadata["delimiter"] == ";"
        assert document.table["Összeg"].tolist() == [1250000.5, 980.25]
        assert document.table["Telefon"].tolist() == ["+36 30 123 4567", "+36 1 234 5678"]
        assert document.metadata["numeric_columns"] == ["Összeg"]

    def test_missing_values_counted(self, tmp_path: Path):
        path = tmp_path / "hianyos.csv"
        path.write_text("a,b\n1,\n,4\n5,6\n", encoding="utf-8")
        assert load_document(path).metadata["missing_values"] == 2

    def test_detect_delimiter(self):
        assert CsvLoader.detect_delimiter("a;b;c\n1;2;3") == ";"
        assert CsvLoader.detect_delimiter("a\tb\n1\t2") == "\t"
        assert CsvLoader.detect_delimiter("egyetlen oszlop\nertek") == ","


class TestPdfLoader:
    def test_pages_and_metadata(self, tmp_path: Path, pdf_factory):
        path = pdf_factory(tmp_path / "doc.pdf", ["Első oldal szövege", "Második oldal"], title="Teszt")
        document = load_document(path)
        assert document.file_type == "pdf"
        assert document.page_count == 2
        assert document.metadata["page_count"] == 2
        assert document.metadata["title"] == "Teszt"
        assert "Első oldal szövege" in document.pages[0]
        assert "Második oldal" in document.text

    def test_page_without_text_is_reported(self, tmp_path: Path, pdf_factory):
        path = pdf_factory(tmp_path / "reszben_ures.pdf", ["Van szöveg", ""])
        assert load_document(path).metadata["pages_without_text"] == [2]

    def test_pdf_without_text_raises(self, tmp_path: Path, pdf_factory):
        path = pdf_factory(tmp_path / "ures.pdf", [""])
        with pytest.raises(EmptyDocumentError):
            load_document(path)

    def test_corrupt_pdf_raises(self, tmp_path: Path):
        path = tmp_path / "hibas.pdf"
        path.write_bytes(b"ez nem egy pdf fajl")
        with pytest.raises(DocumentLoadError):
            load_document(path)

    def test_clean_text_joins_hyphenated_words(self):
        assert PdfLoader.clean_text("doku-\nmentum   feldolgozás\n\n\n\nvége") == "dokumentum feldolgozás\n\nvége"


class TestRegistry:
    def test_supported_extensions(self):
        assert {".txt", ".csv", ".pdf"} <= set(supported_extensions())

    def test_get_loader_by_extension(self):
        assert isinstance(get_loader("x.TXT"), TxtLoader)
        assert isinstance(get_loader("x.csv"), CsvLoader)
        assert isinstance(get_loader("x.pdf"), PdfLoader)

    def test_unsupported_format(self, tmp_path: Path):
        path = tmp_path / "kep.png"
        path.write_bytes(b"\x89PNG")
        with pytest.raises(UnsupportedFormatError):
            load_document(path)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(DocumentLoadError):
            load_document(tmp_path / "nincs.txt")

    def test_register_custom_loader(self, tmp_path: Path):
        @register_loader("upper")
        class UpperLoader(BaseLoader):
            def _load(self, path: Path) -> Document:
                return Document(path=path, text=path.read_text(encoding="utf-8").upper(), file_type="upper")

        try:
            path = tmp_path / "minta.upper"
            path.write_text("kis betűk", encoding="utf-8")
            assert UpperLoader.extensions == (".upper",)
            assert load_document(path).text == "KIS BETŰK"
        finally:
            unregister_loader(".upper")
        assert ".upper" not in supported_extensions()

    def test_register_rejects_non_loader(self):
        with pytest.raises(TypeError):
            register_loader(".x")(dict)


class TestDirectoryIteration:
    def test_iter_files_recursive(self, sample_dir: Path):
        names = [path.name for path in iter_files(sample_dir)]
        assert sorted(names) == ["adatok.csv", "jegyzet.txt", "jelentes.pdf", "level.txt"]

    def test_iter_files_not_recursive(self, sample_dir: Path):
        names = {path.name for path in iter_files(sample_dir, recursive=False)}
        assert "jegyzet.txt" not in names

    def test_iter_files_skips_hidden(self, sample_dir: Path):
        hidden = sample_dir / ".rejtett"
        hidden.mkdir()
        (hidden / "titok.txt").write_text("rejtett", encoding="utf-8")
        assert "titok.txt" not in {path.name for path in iter_files(sample_dir)}

    def test_iter_documents_is_lazy_generator(self, sample_dir: Path):
        documents = iter_documents(sample_dir)
        assert not isinstance(documents, list)
        assert isinstance(next(documents), Document)

    def test_iter_documents_skips_errors(self, sample_dir: Path):
        (sample_dir / "ures.txt").write_text("", encoding="utf-8")
        names = {document.name for document in iter_documents(sample_dir)}
        assert "ures.txt" not in names
        assert len(names) == 4

    def test_iter_documents_raises_when_requested(self, sample_dir: Path):
        (sample_dir / "ures.txt").write_text("", encoding="utf-8")
        with pytest.raises(EmptyDocumentError):
            list(iter_documents(sample_dir, skip_errors=False))

    def test_iter_documents_single_file(self, sample_dir: Path):
        documents = list(iter_documents(sample_dir / "level.txt"))
        assert [document.name for document in documents] == ["level.txt"]

    def test_iter_documents_missing_path(self, tmp_path: Path):
        with pytest.raises(DocumentLoadError):
            list(iter_documents(tmp_path / "nincs_ilyen"))


class TestDocument:
    def test_summary_and_preview(self, tmp_path: Path):
        path = tmp_path / "rovid.txt"
        path.write_text("egy ketto harom\nnegy ot", encoding="utf-8")
        document = load_document(path)
        summary = document.summary()
        assert summary["szavak"] == 5
        assert summary["sorok"] == 2
        assert summary["oldalak"] == 1
        assert document.preview(10) == "egy ketto..."
        assert document.preview(100) == "egy ketto harom negy ot"


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="a demó dokumentumok nincsenek legenerálva")
def test_all_samples_load():
    documents = list(iter_documents(SAMPLES_DIR, skip_errors=False))
    assert len(documents) >= 7
    assert {document.file_type for document in documents} == {"pdf", "txt", "csv"}
