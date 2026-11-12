from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from docproc.document import Document
from docproc.preprocessing import load_stopwords

RANDOM_STATE = 42
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

MODEL_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logreg": "Logisztikus regresszió",
    "linear_svm": "Lineáris SVM",
    "random_forest": "Random Forest",
    "knn": "k-legközelebbi szomszéd",
}
VECTORIZER_NAMES = {"bow": "Bag-of-Words", "tfidf": "TF-IDF"}

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "naive_bayes": {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "classifier__alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
    },
    "logreg": {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [1.0, 10.0, 30.0, 100.0],
    },
    "linear_svm": {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.1, 0.3, 1.0, 3.0],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 500],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "knn": {
        "classifier__n_neighbors": [5, 15, 30],
        "classifier__weights": ["uniform", "distance"],
    },
}


def make_model(name: str) -> Any:
    factories = {
        "naive_bayes": lambda: MultinomialNB(),
        "logreg": lambda: LogisticRegression(max_iter=3000),
        "linear_svm": lambda: LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "knn": lambda: KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance"),
    }
    if name not in factories:
        raise ValueError(f"Ismeretlen modell: {name} (lehetséges: {', '.join(factories)})")
    return factories[name]()


def stop_words_for(language: str | None) -> str | list[str] | None:
    if language == "en":
        return "english"
    if language == "hu":
        return sorted(load_stopwords("hu"))
    return None


def make_vectorizer(
    kind: str = "tfidf",
    language: str | None = "en",
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_df: float = 0.8,
    max_features: int | None = None,
) -> CountVectorizer:
    options = {
        "stop_words": stop_words_for(language),
        "token_pattern": TOKEN_PATTERN,
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **options)
    if kind == "bow":
        return CountVectorizer(**options)
    raise ValueError(f"Ismeretlen vektorizáló: {kind} (lehetséges: {', '.join(VECTORIZER_NAMES)})")


def build_pipeline(model: str, vectorizer: str = "tfidf", language: str | None = "en", **vectorizer_options) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", make_vectorizer(vectorizer, language, **vectorizer_options)),
            ("classifier", make_model(model)),
        ]
    )


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return train_test_split(
        frame["text"],
        frame["label"],
        test_size=test_size,
        stratify=frame["label"],
        random_state=random_state,
    )


def make_folds(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


@dataclass
class EvaluationResult:
    model: str
    vectorizer: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    train_seconds: float
    predict_seconds: float
    labels: list[str] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("labels", "confusion", "report"):
            data.pop(key)
        return data

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.report).T.round(3)

    def confusion_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.confusion, index=self.labels, columns=self.labels)


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    model: str = "",
    vectorizer: str = "",
    train_seconds: float = 0.0,
    predict_seconds: float = 0.0,
) -> EvaluationResult:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return EvaluationResult(
        model=model,
        vectorizer=vectorizer,
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        precision_macro=round(float(precision), 4),
        recall_macro=round(float(recall), 4),
        f1_macro=round(float(f1), 4),
        train_seconds=round(train_seconds, 3),
        predict_seconds=round(predict_seconds, 3),
        labels=labels,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        report=classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    )


def train_and_evaluate(
    pipeline: Pipeline,
    x_train: Sequence[str],
    y_train: Sequence[str],
    x_test: Sequence[str],
    y_test: Sequence[str],
    model: str = "",
    vectorizer: str = "",
) -> EvaluationResult:
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = pipeline.predict(x_test)
    predict_seconds = time.perf_counter() - started
    return evaluate_predictions(
        y_test, predictions, labels=list(pipeline.classes_), model=model, vectorizer=vectorizer,
        train_seconds=train_seconds, predict_seconds=predict_seconds,
    )


def compare_models(
    x: Sequence[str],
    y: Sequence[str],
    models: Iterable[str] = tuple(MODEL_NAMES),
    vectorizers: Iterable[str] = tuple(VECTORIZER_NAMES),
    cv: int = 5,
    language: str | None = "en",
    n_jobs: int = -1,
) -> pd.DataFrame:
    rows = []
    folds = make_folds(cv)
    for model in models:
        for vectorizer in vectorizers:
            scores = cross_validate(
                build_pipeline(model, vectorizer, language),
                x,
                y,
                cv=folds,
                scoring=("accuracy", "f1_macro"),
                n_jobs=n_jobs,
            )
            rows.append(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "modell": MODEL_NAMES[model],
                    "vektorizáló": VECTORIZER_NAMES[vectorizer],
                    "f1_mean": round(float(np.mean(scores["test_f1_macro"])), 4),
                    "f1_std": round(float(np.std(scores["test_f1_macro"])), 4),
                    "accuracy_mean": round(float(np.mean(scores["test_accuracy"])), 4),
                    "accuracy_std": round(float(np.std(scores["test_accuracy"])), 4),
                    "fit_seconds": round(float(np.mean(scores["fit_time"])), 3),
                }
            )
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def tune_model(
    model: str,
    x: Sequence[str],
    y: Sequence[str],
    vectorizer: str = "tfidf",
    param_grid: dict[str, list] | None = None,
    cv: int = 5,
    language: str | None = "en",
    n_jobs: int = -1,
) -> GridSearchCV:
    search = GridSearchCV(
        build_pipeline(model, vectorizer, language),
        param_grid or DEFAULT_PARAM_GRIDS[model],
        scoring="f1_macro",
        cv=make_folds(cv),
        n_jobs=n_jobs,
    )
    search.fit(x, y)
    return search


def tuning_results(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    params = pd.json_normalize(results["params"]).rename(columns=lambda name: name.split("__")[-1])
    frame = pd.concat(
        [params, results[["mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]]],
        axis=1,
    )
    return frame.sort_values("rank_test_score").reset_index(drop=True).round(4)


def calibrate(pipeline: Pipeline, cv: int = 5) -> Pipeline:
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        return pipeline
    calibrated = CalibratedClassifierCV(pipeline.named_steps["classifier"], cv=cv)
    return Pipeline([("vectorizer", pipeline.named_steps["vectorizer"]), ("classifier", calibrated)])


def _linear_weights(classifier: Any) -> np.ndarray | None:
    if isinstance(classifier, CalibratedClassifierCV):
        coefficients = [calibrated.estimator.coef_ for calibrated in classifier.calibrated_classifiers_]
        return np.mean(coefficients, axis=0) if all(c is not None for c in coefficients) else None
    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_)
    if hasattr(classifier, "feature_log_prob_"):
        log_prob = classifier.feature_log_prob_
        return log_prob - log_prob.mean(axis=0)
    return None


def top_features(pipeline: Pipeline, n: int = 10) -> dict[str, list[tuple[str, float]]]:
    names = pipeline.named_steps["vectorizer"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    weights = _linear_weights(classifier)
    if weights is not None:
        labels = list(classifier.classes_)
        if weights.shape[0] == 1 and len(labels) == 2:
            weights = np.vstack([-weights[0], weights[0]])
        return {
            label: [(str(names[index]), round(float(row[index]), 4)) for index in np.argsort(row)[::-1][:n]]
            for label, row in zip(labels, weights)
        }
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        top = np.argsort(importances)[::-1][:n]
        return {"összes kategória": [(str(names[index]), round(float(importances[index]), 4)) for index in top]}
    return {}


def class_scores(pipeline: Pipeline, texts: Sequence[str]) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(texts)
        except AttributeError:
            pass
    decision = np.atleast_2d(pipeline.decision_function(texts))
    exponent = np.exp(decision - decision.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def misclassified_examples(
    pipeline: Pipeline,
    x_test: Sequence[str],
    y_test: Sequence[str],
    limit: int = 20,
    excerpt_length: int = 200,
) -> pd.DataFrame:
    texts = list(x_test)
    truth = list(y_test)
    scores = class_scores(pipeline, texts)
    labels = list(pipeline.classes_)
    rows = []
    for text, actual, row in zip(texts, truth, scores):
        predicted = labels[int(np.argmax(row))]
        if predicted != actual:
            excerpt = " ".join(text.split())[:excerpt_length]
            rows.append({"valós": actual, "becsült": predicted, "magabiztosság": round(float(row.max()), 3), "szöveg": excerpt})
    frame = pd.DataFrame(rows, columns=["valós", "becsült", "magabiztosság", "szöveg"])
    return frame.sort_values("magabiztosság", ascending=False).head(limit).reset_index(drop=True)


@dataclass
class DocumentClassifier:
    pipeline: Pipeline
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return list(self.pipeline.classes_)

    def predict(self, texts: Sequence[str]) -> list[str]:
        return list(self.pipeline.predict(list(texts)))

    def predict_scores(self, texts: Sequence[str], top_k: int = 3) -> list[list[tuple[str, float]]]:
        scores = class_scores(self.pipeline, list(texts))
        labels = self.labels
        return [
            [(labels[index], round(float(row[index]), 4)) for index in np.argsort(row)[::-1][:top_k]]
            for row in scores
        ]

    def classify(self, document: Document, top_k: int = 3) -> list[tuple[str, float]]:
        return self.predict_scores([document.text], top_k=top_k)[0]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": self.metadata}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> DocumentClassifier:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"A modellfájl nem található: {path}. Előbb futtasd a tanítást: python scripts/train.py"
            )
        payload = joblib.load(path)
        return cls(pipeline=payload["pipeline"], metadata=payload.get("metadata", {}))
