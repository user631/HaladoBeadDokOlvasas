from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from docproc import iter_documents, load_document
from docproc.preprocessing import TextPreprocessor
from docproc.statistics import (
    STAT_LABELS,
    compute_statistics,
    corpus_statistics,
    corpus_word_frequencies,
    describe_table,
    document_frequencies,
    document_statistics,
    lix_index,
    readability_level,
    sentence_lengths,
    word_frequencies,
    word_length_distribution,
)
from docproc import visualization as viz

TEXT = (
    "A bérlő a havi bérleti díjat minden hónap 10. napjáig fizeti meg. "
    "A bérleti díj 425 000 Ft.\n\n"
    "A bérlő köteles a helyiséget megóvni."
)


class TestTextStatistics:
    def test_counts(self):
        stats = compute_statistics(TEXT, TextPreprocessor(language="hu"))
        assert stats.language == "hu"
        assert stats.sentence_count == 3
        assert stats.paragraph_count == 2
        assert stats.word_count == 21
        assert stats.char_count == len(TEXT)
        assert stats.number_count == 3
        assert stats.unique_word_count < stats.word_count
        assert 0 < stats.lexical_diversity <= 1

    def test_top_words(self):
        stats = compute_statistics(TEXT, TextPreprocessor(language="hu"), top_n=2)
        assert stats.top_words == [("bérlő", 2), ("bérleti", 2)]

    def test_empty_text(self):
        stats = compute_statistics("", TextPreprocessor(language="hu"))
        assert stats.word_count == 0
        assert stats.avg_word_length == 0
        assert stats.lix == 0
        assert stats.top_words == []

    def test_to_dict_and_labels(self):
        stats = compute_statistics(TEXT)
        assert "top_words" not in stats.to_dict()
        assert "top_words" in stats.to_dict(include_top_words=True)
        assert set(stats.labeled()) == set(STAT_LABELS.values())


class TestHelpers:
    def test_lix_index(self):
        assert lix_index(word_count=100, sentence_count=10, long_word_count=30) == pytest.approx(40.0)
        assert lix_index(0, 0, 0) == 0.0

    @pytest.mark.parametrize(
        ("lix", "label"),
        [(20, "nagyon könnyű"), (35, "könnyű"), (45, "közepes"), (55, "nehéz"), (75, "nagyon nehéz")],
    )
    def test_readability_level(self, lix: float, label: str):
        assert readability_level(lix) == label

    def test_word_frequencies(self):
        assert word_frequencies(["a", "b", "a", "c", "a", "b"], top_n=2) == [("a", 3), ("b", 2)]

    def test_word_length_distribution(self):
        assert word_length_distribution(["ab", "cd", "efg"]) == {2: 2, 3: 1}

    def test_sentence_lengths(self):
        analysis = TextPreprocessor().analyze("Egy két három. Négy öt.")
        assert sentence_lengths(analysis) == [3, 2]


class TestCorpus:
    def test_corpus_statistics(self, sample_dir: Path):
        documents = list(iter_documents(sample_dir))
        stats = corpus_statistics(documents)
        assert isinstance(stats, pd.DataFrame)
        assert set(stats.index) == {document.name for document in documents}
        assert {"file_type", "word_count", "lix"} <= set(stats.columns)

    def test_corpus_statistics_empty(self):
        assert corpus_statistics([]).empty

    def test_corpus_and_document_frequencies(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("számla számla szerződés", encoding="utf-8")
        (tmp_path / "b.txt").write_text("számla ajánlat", encoding="utf-8")
        documents = list(iter_documents(tmp_path))
        preprocessor = TextPreprocessor(language="hu")
        assert corpus_word_frequencies(documents, preprocessor, top_n=1) == [("számla", 3)]
        assert document_frequencies(documents, preprocessor, top_n=1) == [("számla", 2)]

    def test_csv_rows_count_as_sentences(self, tmp_path: Path):
        path = tmp_path / "t.csv"
        path.write_text("nev,osszeg\nAnna,100\nBéla,200\n", encoding="utf-8")
        assert document_statistics(load_document(path)).sentence_count == 3

    def test_describe_table(self, tmp_path: Path):
        path = tmp_path / "t.csv"
        path.write_text("nev,osszeg\nAnna,100\nBéla,300\n", encoding="utf-8")
        description = describe_table(load_document(path))
        assert list(description.index) == ["osszeg"]
        assert description.loc["osszeg", "mean"] == 200

    def test_describe_table_for_text_document(self, sample_dir: Path):
        assert describe_table(load_document(sample_dir / "level.txt")) is None


class TestVisualization:
    def test_figures_are_created_and_saved(self, sample_dir: Path, tmp_path: Path):
        documents = list(iter_documents(sample_dir))
        stats = corpus_statistics(documents)
        analysis = TextPreprocessor().analyze(TEXT)
        figures = [
            viz.plot_top_words([("bérlő", 3), ("díj", 2)]),
            viz.plot_word_length_distribution(analysis.words),
            viz.plot_sentence_length_distribution(sentence_lengths(analysis)),
            viz.plot_corpus_overview(stats),
            viz.plot_readability(stats),
        ]
        for index, figure in enumerate(figures):
            assert isinstance(figure, Figure)
            path = viz.save_figure(figure, tmp_path / "abrak" / f"{index}.png")
            assert path.exists() and path.stat().st_size > 0

    def test_empty_inputs_do_not_fail(self):
        assert isinstance(viz.plot_word_length_distribution([]), Figure)
        assert isinstance(viz.plot_sentence_length_distribution([]), Figure)
        assert isinstance(viz.plot_top_words([]), Figure)
