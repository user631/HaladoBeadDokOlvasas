from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docproc import visualization as viz
from docproc.classification import (
    DEFAULT_PARAM_GRIDS,
    MODEL_NAMES,
    VECTORIZER_NAMES,
    DocumentClassifier,
    build_pipeline,
    calibrate,
    compare_models,
    misclassified_examples,
    split_dataset,
    top_features,
    train_and_evaluate,
    tune_model,
    tuning_results,
)
from docproc.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from docproc.datasets import dataset_summary, load_folder_dataset, load_newsgroups

RESULTS_DIR = REPORTS_DIR / "results"
QUICK_GRIDS = {
    "naive_bayes": {"classifier__alpha": [0.03, 0.3]},
    "logreg": {"classifier__C": [10.0, 30.0]},
    "linear_svm": {"classifier__C": [0.3, 1.0]},
    "random_forest": {"classifier__n_estimators": [200]},
    "knn": {"classifier__n_neighbors": [15]},
}


def step(message: str) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] {message}", flush=True)


def save_csv(frame: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = RESULTS_DIR / name
    frame.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"  mentve: {path.relative_to(ROOT)}")
    return path


def save_json(payload: dict, name: str) -> Path:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  mentve: {path.relative_to(ROOT)}")
    return path


def save_figure(figure, name: str, enabled: bool) -> None:
    if enabled:
        path = viz.save_figure(figure, FIGURES_DIR / name)
        print(f"  ábra: {path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dokumentum-kategorizáló modellek tanítása és összehasonlítása.")
    parser.add_argument("--data-dir", type=Path, help="saját adathalmaz mappája (almappa = kategória); alapértelmezés: 20 Newsgroups")
    parser.add_argument("--language", choices=("en", "hu"), default=None, help="a stopszólista nyelve")
    parser.add_argument("--cv", type=int, default=5, help="keresztvalidációs részhalmazok száma")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_NAMES), default=list(MODEL_NAMES))
    parser.add_argument("--tune-top", type=int, default=2, help="a legjobb N modell hiperparaméter-hangolása")
    parser.add_argument("--quick", action="store_true", help="gyors futás kisebb kereséssel (fejlesztéshez)")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "kategorizalo.joblib")
    parser.add_argument("--no-figures", action="store_true", help="ábrák mentésének kihagyása")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = not args.no_figures
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    step("1. Adatok betöltése és előkészítése")
    if args.data_dir:
        frame = load_folder_dataset(args.data_dir)
        dataset_name = f"saját adathalmaz ({args.data_dir})"
        language = args.language or "hu"
    else:
        frame = load_newsgroups()
        dataset_name = "20 Newsgroups (6 kategória)"
        language = args.language or "en"
    summary = dataset_summary(frame)
    print(f"  {dataset_name}: {len(frame)} dokumentum, {frame['label'].nunique()} kategória")
    print(summary.to_string())
    save_csv(summary, "adathalmaz_osszegzes.csv", index=True)
    save_figure(viz.plot_class_distribution(summary), "ml_osztalyeloszlas.png", figures)
    save_figure(viz.plot_document_lengths(frame), "ml_dokumentumhossz.png", figures)

    x_train, x_test, y_train, y_test = split_dataset(frame)
    print(f"  tanítóhalmaz: {len(x_train)}, teszthalmaz: {len(x_test)} (rétegzett 80/20 felosztás)")

    step("2. Alapmodell: TF-IDF + Multinomial Naive Bayes")
    baseline = train_and_evaluate(build_pipeline("naive_bayes", "tfidf", language), x_train, y_train, x_test, y_test, "naive_bayes", "tfidf")
    print(f"  pontosság: {baseline.accuracy:.4f}, makro F1: {baseline.f1_macro:.4f}")
    save_json(baseline.summary(), "alapmodell.json")

    cv = 3 if args.quick else args.cv
    step(f"3. Modellek összehasonlítása {cv}-szoros keresztvalidációval a tanítóhalmazon")
    comparison = compare_models(x_train, y_train, models=args.models, cv=cv, language=language)
    print(comparison[["modell", "vektorizáló", "f1_mean", "f1_std", "accuracy_mean", "fit_seconds"]].to_string(index=False))
    save_csv(comparison, "modell_osszehasonlitas.csv")
    save_figure(viz.plot_model_comparison(comparison, title=f"Modellek összehasonlítása ({cv}-szoros keresztvalidáció)"), "ml_modell_osszehasonlitas.png", figures)

    step("4. A modellek teljesítménye a teszthalmazon (alapértelmezett paraméterekkel)")
    test_rows = []
    for row in comparison.drop_duplicates("model").itertuples():
        result = train_and_evaluate(build_pipeline(row.model, row.vectorizer, language), x_train, y_train, x_test, y_test, row.model, row.vectorizer)
        test_rows.append({"modell": MODEL_NAMES[row.model], "vektorizáló": VECTORIZER_NAMES[row.vectorizer], "cv_f1": row.f1_mean, **result.summary()})
    test_table = pd.DataFrame(test_rows)
    print(test_table[["modell", "vektorizáló", "cv_f1", "f1_macro", "accuracy", "train_seconds"]].to_string(index=False))
    save_csv(test_table, "teszthalmaz_eredmenyek.csv")

    step(f"5. Hiperparaméter-hangolás (GridSearchCV) a legjobb {args.tune_top} modellre")
    best_configs = comparison.drop_duplicates("model").head(args.tune_top)
    tuned = []
    for row in best_configs.itertuples():
        grid = QUICK_GRIDS[row.model] if args.quick else DEFAULT_PARAM_GRIDS[row.model]
        search = tune_model(row.model, x_train, y_train, row.vectorizer, grid, cv=cv, language=language)
        results = tuning_results(search)
        save_csv(results, f"hangolas_{row.model}.csv")
        print(f"  {MODEL_NAMES[row.model]} ({VECTORIZER_NAMES[row.vectorizer]}): legjobb F1 = {search.best_score_:.4f}, paraméterek: {search.best_params_}")
        tuned.append((search.best_score_, row.model, row.vectorizer, search))
    best_score, best_model, best_vectorizer, best_search = max(tuned, key=lambda item: item[0])

    step("6. Végső modell kiértékelése a teszthalmazon")
    final_pipeline = calibrate(best_search.best_estimator_, cv=cv)
    final = train_and_evaluate(final_pipeline, x_train, y_train, x_test, y_test, best_model, best_vectorizer)
    print(f"  {MODEL_NAMES[best_model]} + {VECTORIZER_NAMES[best_vectorizer]}")
    print(f"  pontosság: {final.accuracy:.4f}, makro precision: {final.precision_macro:.4f}, "
          f"makro recall: {final.recall_macro:.4f}, makro F1: {final.f1_macro:.4f}")
    print(f"  javulás az alapmodellhez képest: {final.f1_macro - baseline.f1_macro:+.4f} makro F1")
    report = final.report_frame()
    print(report.to_string())
    save_csv(report, "vegso_modell_riport.csv", index=True)
    save_csv(final.confusion_frame(), "vegso_modell_confusion.csv", index=True)
    save_figure(viz.plot_confusion_matrix(final.confusion_frame()), "ml_confusion_matrix.png", figures)

    features = top_features(final_pipeline, n=10)
    feature_rows = [{"kategória": label, "helyezés": rank, "szó": word, "súly": weight}
                    for label, items in features.items() for rank, (word, weight) in enumerate(items, start=1)]
    save_csv(pd.DataFrame(feature_rows), "legjellemzobb_szavak.csv")
    save_figure(viz.plot_top_features(features), "ml_legjellemzobb_szavak.png", figures)

    errors = misclassified_examples(final_pipeline, x_test, y_test, limit=25)
    print(f"  hibás besorolások a teszthalmazon: {sum(1 for t, p in zip(y_test, final_pipeline.predict(x_test)) if t != p)} / {len(y_test)}")
    save_csv(errors, "hibas_besorolasok.csv")

    step("7. Modell mentése")
    metadata = {
        "adathalmaz": dataset_name,
        "nyelv": language,
        "modell": best_model,
        "modell_neve": MODEL_NAMES[best_model],
        "vektorizáló": best_vectorizer,
        "paraméterek": {key: str(value) for key, value in best_search.best_params_.items()},
        "cv_f1": round(float(best_score), 4),
        "kategóriák": final.labels,
        "tanítóhalmaz": len(x_train),
        "teszthalmaz": len(x_test),
        "teszt_metrikák": {key: final.summary()[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
        "alapmodell_f1": baseline.f1_macro,
        "tanítás_ideje": datetime.now().isoformat(timespec="seconds"),
    }
    path = DocumentClassifier(final_pipeline, metadata).save(args.model_path)
    print(f"  modell: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    save_json({key: value for key, value in metadata.items() if key != "tanítás_ideje"}, "vegso_modell.json")
    print(f"\nKész ({time.perf_counter() - started:.0f} mp).")


if __name__ == "__main__":
    main()
