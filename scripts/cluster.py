from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docproc import visualization as viz
from docproc.clustering import (
    best_k,
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
from docproc.config import FIGURES_DIR, REPORTS_DIR
from docproc.datasets import load_folder_dataset, load_newsgroups

RESULTS_DIR = REPORTS_DIR / "results"


def step(message: str) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] {message}", flush=True)


def save_csv(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    path = RESULTS_DIR / name
    frame.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"  mentve: {path.relative_to(ROOT)}")


def save_figure(figure, name: str) -> None:
    path = viz.save_figure(figure, FIGURES_DIR / name)
    print(f"  ábra: {path.relative_to(ROOT)}")


def terms_frame(terms: dict[int, list[tuple[str, float]]], prefix: str, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: f"{prefix}{group}", "helyezés": rank, "szó": word, "súly": weight}
         for group, items in terms.items() for rank, (word, weight) in enumerate(items, start=1)]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Felügyelet nélküli csoportosítás: K-Means klaszterezés és NMF témamodellezés.")
    parser.add_argument("--data-dir", type=Path, help="saját címkézett adathalmaz (almappa = kategória)")
    parser.add_argument("--language", choices=("en", "hu"), default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    step("1. Adatok betöltése és TF-IDF vektorizálás")
    if args.data_dir:
        frame = load_folder_dataset(args.data_dir)
        language = args.language or "hu"
    else:
        frame = load_newsgroups()
        language = args.language or "en"
    texts, labels = frame["text"].tolist(), frame["label"].tolist()
    true_k = frame["label"].nunique()
    vectorizer, matrix = vectorize(texts, language)
    names = vectorizer.get_feature_names_out()
    _, features = reduce_dimensions(matrix, n_components=100)
    print(f"  {matrix.shape[0]} dokumentum, {matrix.shape[1]} szó; LSA (TruncatedSVD) után {features.shape[1]} dimenzió")
    print("  a valós címkéket csak a kiértékeléshez használjuk, a klaszterezés nem látja őket")

    step(f"2. A klaszterszám vizsgálata (k = {args.k_min}..{args.k_max})")
    scores = evaluate_k(features, range(args.k_min, args.k_max + 1))
    silhouette_k = best_k(scores)
    print(scores.to_string(index=False))
    print(f"  legjobb sziluett-érték: k = {silhouette_k}; a valós kategóriák száma: {true_k}")
    save_csv(scores, "klaszter_k_valasztas.csv")
    save_figure(viz.plot_k_selection(scores, chosen_k=true_k), "ml_klaszter_k_valasztas.png")

    step("3. K-Means és NMF futtatása, összevetés a valós kategóriákkal")
    rows = []
    kmeans_true = run_kmeans(features, true_k)
    kmeans_best = kmeans_true if silhouette_k == true_k else run_kmeans(features, silhouette_k)
    nmf, document_topics = run_nmf(matrix, true_k)
    nmf_labels = document_topics.argmax(axis=1)
    for method, k, predicted in (
        (f"K-Means (k = {true_k}, a kategóriák száma)", true_k, kmeans_true.labels_),
        (f"K-Means (k = {silhouette_k}, legjobb sziluett)", silhouette_k, kmeans_best.labels_),
        (f"NMF témamodell ({true_k} téma)", true_k, nmf_labels),
    ):
        rows.append({"módszer": method, "k": k, "silhouette": round(silhouette(features, predicted), 4), **external_scores(labels, predicted)})
    evaluation = pd.DataFrame(rows).drop_duplicates("módszer")
    print(evaluation.to_string(index=False))
    save_csv(evaluation, "klaszterezes_ertekeles.csv")

    step("4. A klaszterek tartalma")
    table = contingency_table(labels, kmeans_true.labels_, prefix="K")
    majority = majority_labels(table)
    terms = cluster_terms(matrix, kmeans_true.labels_, names, n=10)
    for cluster, items in terms.items():
        label, share = majority[f"K{cluster}"]
        print(f"  K{cluster}: {label} ({share:.0%}) – {', '.join(word for word, _ in items[:8])}")
    save_csv(table, "klaszter_kontingencia.csv", index=True)
    save_csv(terms_frame(terms, "K", "klaszter"), "klaszter_kulcsszavak.csv")
    save_figure(
        viz.plot_share_heatmap(
            table,
            "Klaszterek és valós kategóriák (K-Means)",
            xlabel="valós kategória",
            ylabel="klaszter",
            colorbar_label="a klaszter arányában",
        ),
        "ml_klaszter_kontingencia.png",
    )

    coordinates = projection_2d(matrix)
    cluster_titles = {cluster: f"K{cluster}: {', '.join(word for word, _ in items[:3])}" for cluster, items in terms.items()}
    save_figure(viz.plot_cluster_map(coordinates, kmeans_true.labels_, cluster_titles, title="K-Means klaszterek két dimenzióban"), "ml_klaszterek_2d.png")
    save_figure(viz.plot_cluster_map(coordinates, labels, title="Valós kategóriák két dimenzióban"), "ml_valos_kategoriak_2d.png")

    step("5. NMF témák")
    topics = topic_terms(nmf, names, n=10)
    topic_table = contingency_table(labels, nmf_labels, prefix="T")
    topic_majority = majority_labels(topic_table)
    for topic, items in topics.items():
        label, share = topic_majority.get(f"T{topic}", ("–", 0.0))
        print(f"  T{topic}: {label} ({share:.0%}) – {', '.join(word for word, _ in items[:8])}")
    save_csv(terms_frame(topics, "T", "téma"), "nmf_temak.csv")
    save_figure(
        viz.plot_top_features({f"T{topic}": items for topic, items in topics.items()}, title="NMF témák legfontosabb szavai"),
        "ml_nmf_temak.png",
    )
    print(f"\nKész ({time.perf_counter() - started:.0f} mp).")


if __name__ == "__main__":
    main()
