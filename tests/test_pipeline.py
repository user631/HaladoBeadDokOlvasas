from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from docproc.classification import DocumentClassifier, build_pipeline
from docproc.exceptions import DocumentLoadError
from docproc.pipeline import DocumentPipeline
from docproc.report import FORMATS, build_html, export_csv, export_html, export_json, export_report

ENGLISH_TEXTS = {
    "hockey.txt": "The team won the hockey game after a great goal in overtime, the season goes on.",
    "car.txt": "The dealer checked the car engine, the brakes and the fuel system before the sale.",
    "match.txt": "Our team played a tough game, the goal came late and the hockey fans celebrated.",
    "engine.txt": "My car needs a new engine and the dealer says the brakes are worn out as well.",
}


@pytest.fixture
def classifier() -> DocumentClassifier:
    texts = ["goal team game season hockey"] * 5 + ["engine car dealer brake fuel"] * 5
    labels = ["sport"] * 5 + ["autók"] * 5
    pipeline = build_pipeline("logreg", "tfidf", min_df=1).fit(texts, labels)
    return DocumentClassifier(pipeline, {"modell_neve": "Logisztikus regresszió", "nyelv": "en", "teszt_metrikák": {"f1_macro": 1.0}})


@pytest.fixture
def english_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "angol"
    directory.mkdir()
    for name, text in ENGLISH_TEXTS.items():
        (directory / name).write_text(text, encoding="utf-8")
    (directory / "ures.txt").write_text("", encoding="utf-8")
    return directory


@pytest.fixture
def hungarian_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "magyar"
    directory.mkdir()
    (directory / "szamla.txt").write_text(
        "Számlaszám: 2026/0142\nKelt: 2026.10.15.\nFizetendő összesen: 138 049 Ft\nE-mail: szamlazas@example.com",
        encoding="utf-8",
    )
    (directory / "ugyfelek.csv").write_text("nev;telefon\nKiss Anna;+36 30 123 4567\n", encoding="utf-8")
    return directory


class TestDocumentPipeline:
    def test_process_english_corpus(self, english_dir: Path, classifier: DocumentClassifier):
        report = DocumentPipeline(classifier=classifier, keyword_count=3).process(english_dir)
        names = {item.name: item for item in report.documents}
        assert set(names) == set(ENGLISH_TEXTS)
        assert names["hockey.txt"].top_category[0] == "sport"
        assert names["engine.txt"].top_category[0] == "autók"
        assert all(len(item.extraction.keywords) <= 3 for item in report.documents)
        assert len(report.errors) == 1 and "ures.txt" in report.errors[0]["fájl"]
        assert report.clustering is not None
        assert names["hockey.txt"].cluster == names["match.txt"].cluster
        assert names["car.txt"].cluster == names["engine.txt"].cluster
        assert names["hockey.txt"].cluster != names["car.txt"].cluster

    def test_language_mismatch_skips_classification(self, hungarian_dir: Path, classifier: DocumentClassifier):
        report = DocumentPipeline(classifier=classifier).process(hungarian_dir)
        szamla = next(item for item in report.documents if item.name == "szamla.txt")
        assert szamla.category is None
        assert "eltér a modell nyelvétől" in szamla.category_note
        assert szamla.extraction.values("amount") == ["138 049 HUF"]
        assert report.clustering is None

    def test_without_model(self, hungarian_dir: Path, tmp_path: Path):
        report = DocumentPipeline(model_path=tmp_path / "nincs.joblib").process(hungarian_dir)
        assert all("nincs betanított modell" in item.category_note for item in report.documents)

    def test_classification_disabled(self, english_dir: Path, classifier: DocumentClassifier):
        report = DocumentPipeline(classifier=classifier, use_classifier=False, use_clustering=False).process(english_dir)
        assert all(item.category is None for item in report.documents)
        assert report.clustering is None

    def test_single_file_and_missing_path(self, hungarian_dir: Path, tmp_path: Path):
        pipeline = DocumentPipeline(use_classifier=False)
        assert [item.name for item in pipeline.process(hungarian_dir / "szamla.txt").documents] == ["szamla.txt"]
        with pytest.raises(DocumentLoadError):
            pipeline.process(tmp_path / "nincs")

    def test_summary_and_frames(self, english_dir: Path, classifier: DocumentClassifier):
        report = DocumentPipeline(classifier=classifier).process(english_dir)
        summary = report.summary()
        assert summary["dokumentumok"] == 4
        assert summary["hibás fájlok"] == 1
        assert summary["típusok"] == {"txt": 4}
        assert summary["kategóriák"] == {"sport": 2, "autók": 2}
        frame = report.statistics_frame()
        assert list(frame["név"]) == sorted(ENGLISH_TEXTS)
        assert {"szavak", "LIX", "kategória", "klaszter", "kulcsszavak"} <= set(frame.columns)
        assert isinstance(report.entities_frame(), pd.DataFrame)


class TestReportExport:
    @pytest.fixture
    def report(self, english_dir: Path, classifier: DocumentClassifier):
        return DocumentPipeline(classifier=classifier).process(english_dir)

    def test_json(self, report, tmp_path: Path):
        payload = json.loads(export_json(report, tmp_path / "r.json").read_text(encoding="utf-8"))
        assert payload["összesítés"]["dokumentumok"] == 4
        assert payload["csoportosítás"]["k"] == report.clustering.k
        first = payload["dokumentumok"][0]
        assert {"név", "statisztika", "kinyert_adatok", "kulcsszavak", "kategória", "klaszter"} <= set(first)
        assert payload["hibák"][0]["fájl"].endswith("ures.txt")

    def test_csv(self, report, tmp_path: Path):
        documents, entities = export_csv(report, tmp_path / "csv")
        assert len(pd.read_csv(documents)) == 4
        assert entities.exists()

    def test_html(self, report, tmp_path: Path):
        html = export_html(report, tmp_path / "r.html").read_text(encoding="utf-8")
        assert html.startswith("<!DOCTYPE html>")
        assert html.count("<details>") == 4
        assert "data:image/png;base64," in html
        assert "Csoportosítás" in html and "Hibás fájlok" in html

    def test_html_escapes_content(self, tmp_path: Path):
        (tmp_path / "x.txt").write_text("Figyelem: <script>alert(1)</script> veszélyes szöveg", encoding="utf-8")
        report = DocumentPipeline(use_classifier=False).process(tmp_path)
        html = build_html(report, include_figures=False)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_export_report_formats(self, report, tmp_path: Path):
        created = export_report(report, tmp_path / "ki", FORMATS)
        assert {path.name for path in created} == {"riport.json", "dokumentumok.csv", "kinyert_adatok.csv", "riport.html"}
        assert [path.name for path in export_report(report, tmp_path / "csak_json", ["json"])] == ["riport.json"]
        with pytest.raises(ValueError):
            export_report(report, tmp_path / "rossz", ["pdf"])
