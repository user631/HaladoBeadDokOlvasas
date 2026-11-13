from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from docproc import visualization as viz
from docproc.clustering import (
    best_k,
    cluster_documents,
    cluster_terms,
    contingency_table,
    evaluate_k,
    external_scores,
    majority_labels,
    projection_2d,
    reduce_dimensions,
    run_kmeans,
    run_nmf,
    silhouette,
    topic_terms,
    vectorize,
)
from tests.test_classification import VOCABULARY, synthetic_dataset


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return synthetic_dataset(per_class=30, seed=1)


@pytest.fixture(scope="module")
def vectors(dataset):
    vectorizer, matrix = vectorize(dataset["text"], min_df=1, max_df=1.0)
    _, features = reduce_dimensions(matrix, n_components=10)
    return vectorizer, matrix, features


class TestBuildingBlocks:
    def test_reduce_dimensions_normalizes(self, vectors):
        _, matrix, features = vectors
        assert features.shape == (matrix.shape[0], 10)
        assert np.allclose(np.linalg.norm(features, axis=1), 1.0)

    def test_reduce_dimensions_caps_components(self):
        _, matrix = vectorize(["egy kettő", "három négy", "öt hat"], language=None, min_df=1, max_df=1.0)
        _, features = reduce_dimensions(matrix, n_components=100)
        assert features.shape[1] == 2

    def test_evaluate_and_best_k(self, vectors):
        _, _, features = vectors
        scores = evaluate_k(features, range(2, 6))
        assert list(scores["k"]) == [2, 3, 4, 5]
        assert scores["inertia"].is_monotonic_decreasing
        assert best_k(scores) == 3

    def test_kmeans_recovers_topics(self, dataset, vectors):
        _, _, features = vectors
        labels = run_kmeans(features, 3).labels_
        scores = external_scores(dataset["label"], labels)
        assert scores["ari"] > 0.9
        assert set(scores) == {"ari", "nmi", "homogeneity", "completeness"}

    def test_silhouette_edge_cases(self, vectors):
        _, _, features = vectors
        assert silhouette(features, [0] * len(features)) == 0.0

    def test_cluster_terms_and_majority(self, dataset, vectors):
        vectorizer, matrix, features = vectors
        labels = run_kmeans(features, 3).labels_
        terms = cluster_terms(matrix, labels, vectorizer.get_feature_names_out(), n=3)
        table = contingency_table(dataset["label"], labels, prefix="K")
        majority = majority_labels(table)
        for cluster, items in terms.items():
            label, share = majority[f"K{cluster}"]
            assert share > 0.9
            assert {word for word, _ in items} <= set(VOCABULARY[label])

    def test_nmf_topics(self, dataset, vectors):
        vectorizer, matrix, _ = vectors
        model, document_topics = run_nmf(matrix, 3)
        assert document_topics.shape == (len(dataset), 3)
        topics = topic_terms(model, vectorizer.get_feature_names_out(), n=4)
        vocabularies = [set(words) for words in VOCABULARY.values()]
        for items in topics.values():
            assert any({word for word, _ in items} <= vocabulary for vocabulary in vocabularies)
        assert external_scores(dataset["label"], document_topics.argmax(axis=1))["ari"] > 0.9

    def test_projection_2d(self, vectors):
        _, matrix, _ = vectors
        assert projection_2d(matrix).shape == (matrix.shape[0], 2)


class TestClusterDocuments:
    def test_automatic_k(self, dataset):
        result = cluster_documents(dataset["text"], k_range=range(2, 6))
        assert result.k == 3
        assert result.k_scores is not None
        assert sum(result.sizes().values()) == len(dataset)
        assert len(result.describe(n_terms=2)) == 3

    def test_fixed_k_small_corpus(self):
        texts = [
            "számla fizetés határidő átutalás összeg",
            "számla összeg átutalás fizetés",
            "bérleti szerződés bérlő bérbeadó díj",
            "szerződés bérlő bérbeadó felmondás",
        ]
        result = cluster_documents(texts, k=2, language="hu")
        assert result.labels[0] == result.labels[1]
        assert result.labels[2] == result.labels[3]
        assert result.labels[0] != result.labels[2]
        assert result.k_scores is None

    def test_emails_are_ignored(self):
        texts = ["anna@example.com számla fizetés", "bela@example.com szerződés bérlő", "cecil@example.com számla fizetés"]
        result = cluster_documents(texts, k=2, language="hu")
        all_terms = {word for items in result.terms.values() for word, _ in items}
        assert "example" not in all_terms and "com" not in all_terms

    @pytest.mark.parametrize(("texts", "k"), [(["a b", "c d"], None), (["alma körte", "alma szilva", "körte szilva"], 3)])
    def test_invalid_input(self, texts, k):
        with pytest.raises(ValueError):
            cluster_documents(texts, k=k, language=None)


class TestClusterPlots:
    def test_plots(self, dataset, vectors):
        _, matrix, features = vectors
        scores = evaluate_k(features, range(2, 5))
        labels = run_kmeans(features, 3).labels_
        table = contingency_table(dataset["label"], labels, prefix="K")
        coordinates = projection_2d(matrix)
        figures = [
            viz.plot_k_selection(scores, chosen_k=3),
            viz.plot_cluster_map(coordinates, labels, {0: "első"}),
            viz.plot_cluster_map(coordinates, dataset["label"]),
            viz.plot_share_heatmap(table, "Teszt"),
        ]
        assert all(isinstance(figure, Figure) for figure in figures)
        assert len([axes for axes in figures[1].axes if axes.get_visible()]) == 3
