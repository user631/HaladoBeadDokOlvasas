from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from docproc.classification import DocumentClassifier
from docproc.clustering import ClusteringResult, cluster_documents
from docproc.config import MODELS_DIR
from docproc.document import Document
from docproc.exceptions import DocProcError, DocumentLoadError
from docproc.extraction import ExtractionResult, InformationExtractor, KeywordExtractor, results_to_frame
from docproc.loaders import iter_files, load_document
from docproc.preprocessing import TextPreprocessor
from docproc.statistics import TextStatistics, corpus_word_frequencies, document_statistics

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = MODELS_DIR / "kategorizalo.joblib"
MIN_DOCUMENTS_FOR_CLUSTERING = 4


@dataclass
class DocumentReport:
    document: Document
    statistics: TextStatistics
    extraction: ExtractionResult
    category: list[tuple[str, float]] | None = None
    category_note: str | None = None
    cluster: int | None = None

    @property
    def name(self) -> str:
        return self.document.name

    @property
    def top_category(self) -> tuple[str, float] | None:
        return self.category[0] if self.category else None

    def to_dict(self) -> dict[str, Any]:
        top = self.top_category
        return {
            "név": self.name,
            "útvonal": str(self.document.path),
            "típus": self.document.file_type,
            "oldalak": self.document.page_count,
            "metaadatok": self.document.metadata,
            "statisztika": self.statistics.to_dict(include_top_words=True),
            "kinyert_adatok": self.extraction.by_kind(),
            "mezők": self.extraction.fields(),
            "kulcsszavak": [{"szó": term, "súly": score} for term, score in self.extraction.keywords],
            "kategória": (
                {"címke": top[0], "valószínűség": top[1], "jelöltek": [{"címke": label, "valószínűség": score} for label, score in self.category]}
                if top else None
            ),
            "kategória_megjegyzés": self.category_note,
            "klaszter": self.cluster,
        }


@dataclass
class CorpusReport:
    source: Path
    documents: list[DocumentReport] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    clustering: ClusteringResult | None = None
    corpus_terms: list[tuple[str, int]] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now().replace(microsecond=0))

    def summary(self) -> dict[str, Any]:
        return {
            "dokumentumok": len(self.documents),
            "hibás fájlok": len(self.errors),
            "típusok": dict(Counter(report.document.file_type for report in self.documents)),
            "nyelvek": dict(Counter(report.statistics.language for report in self.documents)),
            "szavak összesen": sum(report.statistics.word_count for report in self.documents),
            "oldalak összesen": sum(report.document.page_count for report in self.documents),
            "kinyert adatok": sum(len(report.extraction.entities) for report in self.documents),
            "kategóriák": dict(Counter(report.top_category[0] for report in self.documents if report.top_category)),
            "klaszterek": self.clustering.k if self.clustering else None,
        }

    def statistics_frame(self) -> pd.DataFrame:
        rows = []
        for report in self.documents:
            stats = report.statistics
            top = report.top_category
            rows.append(
                {
                    "név": report.name,
                    "típus": report.document.file_type,
                    "nyelv": stats.language,
                    "oldalak": report.document.page_count,
                    "szavak": stats.word_count,
                    "mondatok": stats.sentence_count,
                    "LIX": stats.lix,
                    "olvashatóság": stats.readability,
                    "kinyert adatok": len(report.extraction.entities),
                    "kategória": top[0] if top else "",
                    "valószínűség": top[1] if top else None,
                    "klaszter": report.cluster,
                    "kulcsszavak": ", ".join(term for term, _ in report.extraction.keywords[:5]),
                }
            )
        return pd.DataFrame(rows)

    def entities_frame(self) -> pd.DataFrame:
        return results_to_frame(report.extraction for report in self.documents)

    def to_dict(self) -> dict[str, Any]:
        clustering = None
        if self.clustering:
            clustering = {
                "k": self.clustering.k,
                "sziluett": self.clustering.silhouette,
                "klaszterek": {
                    str(cluster): {
                        "kulcsszavak": [word for word, _ in terms],
                        "dokumentumok": [report.name for report in self.documents if report.cluster == cluster],
                    }
                    for cluster, terms in self.clustering.terms.items()
                },
            }
        return {
            "forrás": str(self.source),
            "készült": self.created_at.isoformat(),
            "összesítés": self.summary(),
            "modell": self.model_info,
            "gyakori_szavak": [{"szó": word, "darab": count} for word, count in self.corpus_terms],
            "csoportosítás": clustering,
            "dokumentumok": [report.to_dict() for report in self.documents],
            "hibák": self.errors,
        }


class DocumentPipeline:
    def __init__(
        self,
        classifier: DocumentClassifier | None = None,
        model_path: Path | None = DEFAULT_MODEL_PATH,
        use_classifier: bool = True,
        extraction_kinds: Iterable[str] | None = None,
        keyword_count: int = 10,
        use_clustering: bool = True,
        preprocessor: TextPreprocessor | None = None,
        top_words: int = 10,
    ):
        self.classifier = classifier
        if self.classifier is None and use_classifier and model_path is not None and Path(model_path).exists():
            self.classifier = DocumentClassifier.load(model_path)
        self.use_classifier = use_classifier
        self.extractor = InformationExtractor(extraction_kinds)
        self.keyword_count = keyword_count
        self.use_clustering = use_clustering
        self.preprocessor = preprocessor or TextPreprocessor()
        self.top_words = top_words

    @property
    def model_language(self) -> str | None:
        return self.classifier.metadata.get("nyelv") if self.classifier else None

    def _classify(self, document: Document, language: str) -> tuple[list[tuple[str, float]] | None, str | None]:
        if not self.use_classifier:
            return None, "a kategorizálás ki van kapcsolva"
        if self.classifier is None:
            return None, "nincs betanított modell (python scripts/train.py)"
        model_language = self.model_language
        if model_language and language not in (model_language, "unknown"):
            return None, f"a dokumentum nyelve ({language}) eltér a modell nyelvétől ({model_language})"
        return self.classifier.classify(document, top_k=3), None

    def process_document(self, document: Document, keyword_extractor: KeywordExtractor | None = None) -> DocumentReport:
        statistics = document_statistics(document, self.preprocessor, top_n=self.top_words)
        extraction = self.extractor.extract(document)
        if self.keyword_count > 0:
            keyword_extractor = keyword_extractor or KeywordExtractor(top_n=self.keyword_count)
            extraction.keywords = keyword_extractor.extract(document.text, top_n=self.keyword_count)
        category, note = self._classify(document, statistics.language)
        return DocumentReport(document, statistics, extraction, category, note)

    def load(self, source: str | Path, recursive: bool = True) -> tuple[list[Document], list[dict[str, str]]]:
        source = Path(source)
        if not source.exists():
            raise DocumentLoadError(source, "a megadott útvonal nem létezik")
        paths = [source] if source.is_file() else list(iter_files(source, recursive=recursive))
        documents, errors = [], []
        for path in paths:
            try:
                documents.append(load_document(path))
            except DocProcError as exc:
                logger.warning("%s", exc)
                errors.append({"fájl": str(path), "hiba": str(exc)})
        return documents, errors

    def _cluster(self, reports: list[DocumentReport]) -> ClusteringResult | None:
        if not self.use_clustering or len(reports) < MIN_DOCUMENTS_FOR_CLUSTERING:
            return None
        languages = Counter(report.statistics.language for report in reports)
        language = languages.most_common(1)[0][0]
        try:
            result = cluster_documents(
                [report.document.text for report in reports],
                k_range=range(2, min(7, len(reports))),
                language=language if language in ("hu", "en") else None,
            )
        except ValueError as exc:
            logger.warning("A csoportosítás kimaradt: %s", exc)
            return None
        for report, label in zip(reports, result.labels):
            report.cluster = int(label)
        return result

    def process(self, source: str | Path, recursive: bool = True) -> CorpusReport:
        documents, errors = self.load(source, recursive)
        keyword_extractor = KeywordExtractor(top_n=self.keyword_count).fit(document.text for document in documents)
        reports = [self.process_document(document, keyword_extractor) for document in documents]
        clustering = self._cluster(reports)
        return CorpusReport(
            source=Path(source),
            documents=reports,
            errors=errors,
            clustering=clustering,
            corpus_terms=corpus_word_frequencies(documents, self.preprocessor, top_n=20),
            model_info=dict(self.classifier.metadata) if self.classifier else {},
        )
