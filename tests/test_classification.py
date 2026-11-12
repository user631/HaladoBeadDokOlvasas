from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from docproc import Document
from docproc.classification import (
    MODEL_NAMES,
    DocumentClassifier,
    build_pipeline,
    calibrate,
    class_scores,
    compare_models,
    evaluate_predictions,
    make_model,
    make_vectorizer,
    misclassified_examples,
    split_dataset,
    top_features,
    train_and_evaluate,
    tune_model,
    tuning_results,
)
from docproc.datasets import clean_newsgroup_text, dataset_summary, load_folder_dataset, prepare_dataset

VOCABULARY = {
    "sport": ["game", "team", "goal", "season", "player", "coach", "score", "league", "hockey", "match"],
    "autók": ["car", "engine", "dealer", "wheel", "brake", "fuel", "drive", "tire", "speed", "motor"],
    "orvostudomány": ["doctor", "patient", "disease", "treatment", "medicine", "clinic", "symptom", "drug", "health", "therapy"],
}
COMMON = ["the", "and", "is", "was", "very", "really", "today", "people", "think", "about"]


def synthetic_dataset(per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label, words in VOCABULARY.items():
        for _ in range(per_class):
            tokens = rng.choices(words, k=12) + rng.choices(COMMON, k=8)
            rng.shuffle(tokens)
            rows.append({"text": " ".join(tokens), "label": label})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return synthetic_dataset()


@pytest.fixture(scope="module")
def split(dataset):
    return split_dataset(dataset)


@pytest.fixture(scope="module")
def trained(split):
    x_train, _, y_train, _ = split
    pipeline = build_pipeline("logreg", "tfidf", min_df=1)
    return pipeline.fit(x_train, y_train)


class TestDatasets:
    def test_clean_newsgroup_text(self):
        text = "> idézett sor\nKérdés: ki ír a foo@bar.com címre?\n=====\nLátogass el ide: https://x.example"
        assert clean_newsgroup_text(text) == "Kérdés: ki ír a címre? Látogass el ide:"

    def test_prepare_dataset_drops_short_and_duplicates(self):
        frame = pd.DataFrame({"text": ["rövid", "x" * 60, "x" * 60, "y" * 70], "label": ["a", "b", "b", "c"]})
        prepared = prepare_dataset(frame)
        assert list(prepared["label"]) == ["b", "c"]

    def test_load_folder_dataset(self, tmp_path: Path):
        for label, text in {"szamla": "Számla a vásárlásról " * 5, "szerzodes": "Bérleti szerződés az irodáról " * 5}.items():
            (tmp_path / label).mkdir()
            (tmp_path / label / "a.txt").write_text(text, encoding="utf-8")
        (tmp_path / "szamla" / "rovid.txt").write_text("rövid", encoding="utf-8")
        frame = load_folder_dataset(tmp_path)
        assert sorted(frame["label"]) == ["szamla", "szerzodes"]

    def test_load_folder_dataset_empty(self, tmp_path: Path):
        with pytest.raises(ValueError):
            load_folder_dataset(tmp_path)

    def test_dataset_summary(self, dataset):
        summary = dataset_summary(dataset)
        assert set(summary.index) == set(VOCABULARY)
        assert summary["documents"].sum() == len(dataset)
        assert summary["share"].sum() == pytest.approx(1.0, abs=0.01)


class TestBuilding:
    def test_all_models_can_be_created(self):
        for name in MODEL_NAMES:
            assert make_model(name) is not None

    def test_unknown_model_and_vectorizer(self):
        with pytest.raises(ValueError):
            make_model("xgboost")
        with pytest.raises(ValueError):
            make_vectorizer("word2vec")

    def test_hungarian_stop_words(self):
        vectorizer = make_vectorizer("bow", language="hu", min_df=1, max_df=1.0)
        vectorizer.fit(["a számla és a szerződés"])
        assert set(vectorizer.get_feature_names_out()) == {"számla", "szerződés"}

    def test_split_is_stratified(self, dataset, split):
        x_train, x_test, y_train, y_test = split
        assert len(x_test) == pytest.approx(len(dataset) * 0.2, abs=1)
        assert set(y_test.value_counts()) == {8}


class TestTrainingAndEvaluation:
    @pytest.mark.parametrize("model", list(MODEL_NAMES))
    def test_models_learn_synthetic_topics(self, split, model):
        x_train, x_test, y_train, y_test = split
        result = train_and_evaluate(build_pipeline(model, "tfidf", min_df=1), x_train, y_train, x_test, y_test, model, "tfidf")
        assert result.f1_macro >= 0.9
        assert result.labels == sorted(VOCABULARY)

    def test_evaluate_predictions(self):
        result = evaluate_predictions(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        assert result.accuracy == 0.75
        assert result.confusion == [[1, 1], [0, 2]]
        assert "f1_macro" in result.summary() and "confusion" not in result.summary()
        assert result.confusion_frame().loc["a", "b"] == 1
        assert "precision" in result.report_frame().columns

    def test_compare_models(self, dataset):
        table = compare_models(dataset["text"], dataset["label"], models=["naive_bayes", "logreg"], vectorizers=["bow", "tfidf"], cv=3, n_jobs=1)
        assert len(table) == 4
        assert table["f1_mean"].is_monotonic_decreasing
        assert {"modell", "vektorizáló", "f1_std", "fit_seconds"} <= set(table.columns)

    def test_tune_model(self, dataset):
        search = tune_model("naive_bayes", dataset["text"], dataset["label"], param_grid={"classifier__alpha": [0.1, 1.0]}, cv=3, n_jobs=1)
        results = tuning_results(search)
        assert list(results.columns[:1]) == ["alpha"]
        assert len(results) == 2
        assert results.loc[0, "rank_test_score"] == 1


class TestInterpretation:
    def test_top_features_linear(self, trained):
        features = top_features(trained, n=3)
        assert set(features) == set(VOCABULARY)
        assert {word for word, _ in features["sport"]} <= set(VOCABULARY["sport"])

    def test_top_features_naive_bayes_and_forest(self, split):
        x_train, _, y_train, _ = split
        nb = build_pipeline("naive_bayes", "bow", min_df=1).fit(x_train, y_train)
        assert {word for word, _ in top_features(nb, 3)["autók"]} <= set(VOCABULARY["autók"])
        forest = build_pipeline("random_forest", "tfidf", min_df=1).fit(x_train, y_train)
        assert list(top_features(forest, 5)) == ["összes kategória"]

    def test_class_scores_sum_to_one(self, split):
        x_train, x_test, y_train, _ = split
        svm = build_pipeline("linear_svm", "tfidf", min_df=1).fit(x_train, y_train)
        scores = class_scores(svm, list(x_test))
        assert scores.shape == (len(x_test), 3)
        assert scores.sum(axis=1) == pytest.approx([1.0] * len(x_test))

    def test_calibration(self, split):
        x_train, x_test, y_train, _ = split
        calibrated = calibrate(build_pipeline("linear_svm", "tfidf", min_df=1), cv=3).fit(x_train, y_train)
        assert hasattr(calibrated, "predict_proba")
        assert set(top_features(calibrated, 2)) == set(VOCABULARY)

    def test_misclassified_examples(self, trained, split):
        _, x_test, _, y_test = split
        wrong_labels = ["sport" if label != "sport" else "autók" for label in y_test]
        frame = misclassified_examples(trained, x_test, wrong_labels, limit=5)
        assert len(frame) == 5
        assert list(frame.columns) == ["valós", "becsült", "magabiztosság", "szöveg"]


class TestDocumentClassifier:
    def test_save_load_and_predict(self, trained, tmp_path: Path):
        classifier = DocumentClassifier(trained, metadata={"model": "logreg"})
        path = classifier.save(tmp_path / "modell" / "teszt.joblib")
        loaded = DocumentClassifier.load(path)
        assert loaded.metadata == {"model": "logreg"}
        assert loaded.labels == sorted(VOCABULARY)
        assert loaded.predict(["the doctor gave the patient a new drug"]) == ["orvostudomány"]
        document = Document(path=Path("x.txt"), text="engine brake fuel car dealer", file_type="txt")
        scores = loaded.classify(document, top_k=2)
        assert scores[0][0] == "autók"
        assert len(scores) == 2
        assert scores[0][1] >= scores[1][1]

    def test_load_missing_model(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DocumentClassifier.load(tmp_path / "nincs.joblib")
