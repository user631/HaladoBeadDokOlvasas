from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from docproc.document import Document
from docproc.preprocessing import TextAnalysis, TextPreprocessor, tokenize

LONG_WORD_MIN_LENGTH = 7
READING_SPEED_WPM = 200
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

LIX_LEVELS = (
    (30.0, "nagyon könnyű"),
    (40.0, "könnyű"),
    (50.0, "közepes"),
    (60.0, "nehéz"),
    (math.inf, "nagyon nehéz"),
)

STAT_LABELS = {
    "language": "nyelv",
    "char_count": "karakterek",
    "char_count_no_spaces": "karakterek (szóköz nélkül)",
    "word_count": "szavak",
    "unique_word_count": "egyedi szavak",
    "content_word_count": "tartalmas szavak",
    "sentence_count": "mondatok",
    "paragraph_count": "bekezdések",
    "number_count": "számok",
    "avg_word_length": "átl. szóhossz",
    "avg_sentence_length": "átl. mondathossz (szó)",
    "lexical_diversity": "lexikai diverzitás",
    "hapax_count": "egyszer előforduló szavak",
    "long_word_ratio": "hosszú szavak aránya",
    "lix": "LIX index",
    "readability": "olvashatóság",
    "reading_time_min": "olvasási idő (perc)",
}


@dataclass
class TextStatistics:
    language: str
    char_count: int
    char_count_no_spaces: int
    word_count: int
    unique_word_count: int
    content_word_count: int
    sentence_count: int
    paragraph_count: int
    number_count: int
    avg_word_length: float
    avg_sentence_length: float
    lexical_diversity: float
    hapax_count: int
    long_word_ratio: float
    lix: float
    readability: str
    reading_time_min: float
    top_words: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self, include_top_words: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_top_words:
            data.pop("top_words")
        return data

    def labeled(self) -> dict[str, Any]:
        return {STAT_LABELS[key]: value for key, value in self.to_dict().items()}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def lix_index(word_count: int, sentence_count: int, long_word_count: int) -> float:
    if word_count == 0:
        return 0.0
    return word_count / max(sentence_count, 1) + 100 * long_word_count / word_count


def readability_level(lix: float) -> str:
    for upper_bound, label in LIX_LEVELS:
        if lix < upper_bound:
            return label
    return LIX_LEVELS[-1][1]


def word_frequencies(tokens: Iterable[str], top_n: int | None = 10) -> list[tuple[str, int]]:
    return Counter(tokens).most_common(top_n)


def word_length_distribution(words: Iterable[str]) -> dict[int, int]:
    return dict(sorted(Counter(len(word) for word in words).items()))


def sentence_lengths(analysis: TextAnalysis) -> list[int]:
    return [len(tokenize(sentence)) for sentence in analysis.sentences]


def statistics_from_analysis(text: str, analysis: TextAnalysis, top_n: int = 10) -> TextStatistics:
    words = analysis.words
    word_count = len(words)
    sentence_count = len(analysis.sentences)
    counts = Counter(words)
    long_word_count = sum(1 for word in words if len(word) >= LONG_WORD_MIN_LENGTH)
    lix = lix_index(word_count, sentence_count, long_word_count)
    return TextStatistics(
        language=analysis.language,
        char_count=len(text),
        char_count_no_spaces=sum(1 for char in text if not char.isspace()),
        word_count=word_count,
        unique_word_count=len(counts),
        content_word_count=len(analysis.content_words),
        sentence_count=sentence_count,
        paragraph_count=len(analysis.paragraphs),
        number_count=len(_NUMBER.findall(text)),
        avg_word_length=round(_safe_ratio(sum(len(word) for word in words), word_count), 2),
        avg_sentence_length=round(_safe_ratio(word_count, sentence_count), 2),
        lexical_diversity=round(_safe_ratio(len(counts), word_count), 3),
        hapax_count=sum(1 for count in counts.values() if count == 1),
        long_word_ratio=round(_safe_ratio(long_word_count, word_count), 3),
        lix=round(lix, 1),
        readability=readability_level(lix),
        reading_time_min=round(word_count / READING_SPEED_WPM, 2),
        top_words=word_frequencies(analysis.content_words, top_n),
    )


def compute_statistics(
    text: str,
    preprocessor: TextPreprocessor | None = None,
    top_n: int = 10,
) -> TextStatistics:
    preprocessor = preprocessor or TextPreprocessor()
    return statistics_from_analysis(text, preprocessor.analyze(text), top_n=top_n)


def document_statistics(
    document: Document,
    preprocessor: TextPreprocessor | None = None,
    top_n: int = 10,
) -> TextStatistics:
    preprocessor = preprocessor or TextPreprocessor()
    analysis = preprocessor.analyze(document.text)
    if document.file_type == "csv":
        analysis.sentences = [line for line in document.text.splitlines() if line.strip()]
    return statistics_from_analysis(document.text, analysis, top_n=top_n)


def corpus_statistics(
    documents: Iterable[Document],
    preprocessor: TextPreprocessor | None = None,
) -> pd.DataFrame:
    preprocessor = preprocessor or TextPreprocessor()
    rows = []
    for document in documents:
        stats = document_statistics(document, preprocessor)
        rows.append({"name": document.name, "file_type": document.file_type, **stats.to_dict()})
    return pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()


def corpus_word_frequencies(
    documents: Iterable[Document],
    preprocessor: TextPreprocessor | None = None,
    top_n: int | None = 20,
) -> list[tuple[str, int]]:
    preprocessor = preprocessor or TextPreprocessor()
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(preprocessor.tokens(document.text))
    return counter.most_common(top_n)


def document_frequencies(
    documents: Iterable[Document],
    preprocessor: TextPreprocessor | None = None,
    top_n: int | None = 20,
) -> list[tuple[str, int]]:
    preprocessor = preprocessor or TextPreprocessor()
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(set(preprocessor.tokens(document.text)))
    return counter.most_common(top_n)


def describe_table(document: Document) -> pd.DataFrame | None:
    if document.table is None:
        return None
    numeric = document.table.select_dtypes("number")
    if numeric.empty:
        return None
    return numeric.describe().T.round(2)
