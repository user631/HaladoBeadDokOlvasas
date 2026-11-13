from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from docproc.classification import RANDOM_STATE, make_vectorizer
from docproc.preprocessing import clean_text

SMALL_CORPUS_SIZE = 30


def vectorize(
    texts: Sequence[str],
    language: str | None = "en",
    min_df: int = 2,
    max_df: float = 0.8,
    max_features: int | None = None,
) -> tuple[TfidfVectorizer, spmatrix]:
    vectorizer = make_vectorizer("tfidf", language, min_df=min_df, max_df=max_df, max_features=max_features)
    return vectorizer, vectorizer.fit_transform(texts)


def reduce_dimensions(matrix: spmatrix, n_components: int = 100) -> tuple[Pipeline, np.ndarray]:
    n_components = max(1, min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1))
    lsa = make_pipeline(TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE), Normalizer(copy=False))
    return lsa, lsa.fit_transform(matrix)


def run_kmeans(features: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(features)


def silhouette(features: np.ndarray, labels: Sequence[int]) -> float:
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return 0.0
    return float(silhouette_score(features, labels, metric="cosine"))


def evaluate_k(features: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = run_kmeans(features, k)
        rows.append({"k": k, "inertia": round(float(model.inertia_), 3), "silhouette": round(silhouette(features, model.labels_), 4)})
    return pd.DataFrame(rows)


def best_k(scores: pd.DataFrame) -> int:
    return int(scores.loc[scores["silhouette"].idxmax(), "k"])


def cluster_terms(
    matrix: spmatrix,
    labels: Sequence[int],
    feature_names: Sequence[str],
    n: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    labels = np.asarray(labels)
    result = {}
    for cluster in sorted(set(labels.tolist())):
        mean = np.asarray(matrix[labels == cluster].mean(axis=0)).ravel()
        top = np.argsort(mean)[::-1][:n]
        result[int(cluster)] = [(str(feature_names[index]), round(float(mean[index]), 4)) for index in top if mean[index] > 0]
    return result


def run_nmf(matrix: spmatrix, n_topics: int) -> tuple[NMF, np.ndarray]:
    model = NMF(n_components=n_topics, init="nndsvda", max_iter=500, random_state=RANDOM_STATE)
    return model, model.fit_transform(matrix)


def topic_terms(model: NMF, feature_names: Sequence[str], n: int = 10) -> dict[int, list[tuple[str, float]]]:
    return {
        topic: [(str(feature_names[index]), round(float(weights[index]), 4)) for index in np.argsort(weights)[::-1][:n]]
        for topic, weights in enumerate(model.components_)
    }


def external_scores(true_labels: Sequence[str], predicted: Sequence[int]) -> dict[str, float]:
    return {
        "ari": round(float(adjusted_rand_score(true_labels, predicted)), 4),
        "nmi": round(float(normalized_mutual_info_score(true_labels, predicted)), 4),
        "homogeneity": round(float(homogeneity_score(true_labels, predicted)), 4),
        "completeness": round(float(completeness_score(true_labels, predicted)), 4),
    }


def contingency_table(true_labels: Sequence[str], predicted: Sequence[int], prefix: str = "") -> pd.DataFrame:
    table = pd.crosstab(pd.Series(list(predicted), name="klaszter"), pd.Series(list(true_labels), name="kategória"))
    table.index = [f"{prefix}{index}" for index in table.index]
    return table


def majority_labels(table: pd.DataFrame) -> dict[str, tuple[str, float]]:
    shares = table.div(table.sum(axis=1), axis=0)
    return {str(index): (str(shares.loc[index].idxmax()), round(float(shares.loc[index].max()), 3)) for index in table.index}


def projection_2d(matrix: spmatrix) -> np.ndarray:
    if matrix.shape[1] < 3:
        dense = matrix.toarray()
        return np.pad(dense, ((0, 0), (0, max(0, 2 - dense.shape[1]))))[:, :2]
    return TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)


@dataclass
class ClusteringResult:
    k: int
    labels: np.ndarray
    silhouette: float
    terms: dict[int, list[tuple[str, float]]]
    coordinates: np.ndarray
    k_scores: pd.DataFrame | None = field(default=None, repr=False)

    def sizes(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    def describe(self, n_terms: int = 5) -> dict[int, str]:
        return {cluster: ", ".join(term for term, _ in terms[:n_terms]) for cluster, terms in self.terms.items()}


def cluster_documents(
    texts: Sequence[str],
    k: int | None = None,
    k_range: Iterable[int] = range(2, 9),
    language: str | None = "en",
    n_terms: int = 8,
) -> ClusteringResult:
    texts = [clean_text(text, lowercase=False) for text in texts]
    if len(texts) < 3:
        raise ValueError("A klaszterezéshez legalább 3 dokumentum szükséges.")
    small = len(texts) < SMALL_CORPUS_SIZE
    vectorizer, matrix = vectorize(texts, language, min_df=1 if small else 2, max_df=1.0 if small else 0.8)
    if matrix.shape[1] < 2:
        raise ValueError("A dokumentumokban túl kevés a közös szó a klaszterezéshez.")
    _, features = reduce_dimensions(matrix)

    scores = None
    if k is None:
        candidates = [value for value in k_range if 2 <= value < len(texts)]
        scores = evaluate_k(features, candidates)
        k = best_k(scores)
    if not 2 <= k < len(texts):
        raise ValueError(f"A klaszterek száma 2 és {len(texts) - 1} között lehet.")

    model = run_kmeans(features, k)
    return ClusteringResult(
        k=k,
        labels=model.labels_,
        silhouette=round(silhouette(features, model.labels_), 4),
        terms=cluster_terms(matrix, model.labels_, vectorizer.get_feature_names_out(), n_terms),
        coordinates=projection_2d(matrix),
        k_scores=scores,
    )
