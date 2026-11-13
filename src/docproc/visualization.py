from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from docproc.statistics import LIX_LEVELS

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FILE_TYPE_COLORS = {"pdf": SERIES[0], "txt": SERIES[1], "csv": SERIES[2]}
FALLBACK_COLOR = "#8a8984"


def _new_figure(width: float, height: float) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    axes = figure.add_subplot()
    _style_axes(axes)
    return figure, axes


def _style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.set_axisbelow(True)


def _set_title(axes: Axes, title: str, subtitle: str | None = None) -> None:
    axes.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=18 if subtitle else 8)
    if subtitle:
        axes.text(0, 1.01, subtitle, transform=axes.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")


def plot_top_words(
    frequencies: Sequence[tuple[str, int]],
    title: str = "Leggyakoribb szavak",
    subtitle: str | None = None,
) -> Figure:
    words = [word for word, _ in frequencies][::-1]
    counts = [count for _, count in frequencies][::-1]
    figure, axes = _new_figure(7, max(2.5, 0.32 * len(words) + 1.2))
    axes.barh(words, counts, height=0.62, color=SERIES[0])
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel("előfordulás")
    axes.margins(y=0.02)
    _set_title(axes, title, subtitle)
    return figure


def plot_word_length_distribution(
    words: Sequence[str],
    title: str = "Szóhosszok eloszlása",
    subtitle: str | None = None,
) -> Figure:
    lengths = pd.Series([len(word) for word in words], dtype="int64")
    counts = lengths.value_counts().sort_index()
    figure, axes = _new_figure(7, 3.6)
    axes.bar(counts.index, counts.values, width=0.8, color=SERIES[0])
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_xlabel("szóhossz (karakter)")
    axes.set_ylabel("szavak száma")
    if not counts.empty:
        mean = lengths.mean()
        axes.set_ylim(top=counts.max() * 1.18)
        axes.axvline(mean, color=TEXT_SECONDARY, linewidth=1.2, linestyle="--")
        axes.text(mean, counts.max() * 1.16, f" átlag: {mean:.1f}", color=TEXT_SECONDARY, fontsize=9, va="top")
    _set_title(axes, title, subtitle)
    return figure


def plot_sentence_length_distribution(
    lengths: Sequence[int],
    title: str = "Mondathosszok eloszlása",
    subtitle: str | None = None,
    bins: int = 15,
) -> Figure:
    figure, axes = _new_figure(7, 3.6)
    if lengths:
        axes.hist(lengths, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_xlabel("mondathossz (szó)")
    axes.set_ylabel("mondatok száma")
    _set_title(axes, title, subtitle)
    return figure


def _type_legend(axes: Axes, file_types: Sequence[str]) -> None:
    present = [file_type for file_type in FILE_TYPE_COLORS if file_type in set(file_types)]
    handles = [Patch(color=FILE_TYPE_COLORS[file_type], label=file_type.upper()) for file_type in present]
    if len(handles) >= 2:
        legend = axes.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)


def plot_corpus_overview(
    stats: pd.DataFrame,
    column: str = "word_count",
    title: str = "Dokumentumok terjedelme",
    xlabel: str = "szavak száma",
    subtitle: str | None = None,
) -> Figure:
    ordered = stats.sort_values(column)
    colors = [FILE_TYPE_COLORS.get(file_type, FALLBACK_COLOR) for file_type in ordered["file_type"]]
    figure, axes = _new_figure(7, max(2.5, 0.4 * len(ordered) + 1.2))
    axes.barh(ordered.index, ordered[column], height=0.62, color=colors)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel(xlabel)
    _type_legend(axes, list(ordered["file_type"]))
    _set_title(axes, title, subtitle)
    return figure


def plot_readability(stats: pd.DataFrame, title: str = "Olvashatóság (LIX index)") -> Figure:
    figure = plot_corpus_overview(
        stats,
        column="lix",
        title=title,
        xlabel="LIX index (magasabb = nehezebb szöveg)",
        subtitle="sávhatárok: 30 / 40 / 50 / 60 (nagyon könnyű → nagyon nehéz)",
    )
    axes = figure.axes[0]
    for boundary, _ in LIX_LEVELS[:-1]:
        axes.axvline(boundary, color=GRID, linewidth=1.2, linestyle="--", zorder=0)
    return figure


def save_figure(figure: Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=SURFACE, metadata={"Software": None})
    return path


def plot_class_distribution(summary: pd.DataFrame, title: str = "Dokumentumok kategóriánként") -> Figure:
    ordered = summary.sort_values("documents")
    figure, axes = _new_figure(7, max(2.5, 0.42 * len(ordered) + 1.2))
    axes.barh(ordered.index, ordered["documents"], height=0.62, color=SERIES[0])
    for label, count, share in zip(ordered.index, ordered["documents"], ordered["share"]):
        axes.text(count, label, f" {count} ({share:.0%})", va="center", fontsize=9, color=TEXT_SECONDARY)
    axes.set_xlim(right=ordered["documents"].max() * 1.18)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel("dokumentumok száma")
    _set_title(axes, title, f"összesen {int(ordered['documents'].sum())} dokumentum")
    return figure


def plot_document_lengths(frame: pd.DataFrame, title: str = "Dokumentumhosszok kategóriánként") -> Figure:
    lengths = frame.assign(words=frame["text"].str.split().str.len())
    order = lengths.groupby("label")["words"].median().sort_values().index.tolist()
    data = [lengths.loc[lengths["label"] == label, "words"].to_numpy() for label in order]
    figure, axes = _new_figure(7, max(2.5, 0.42 * len(order) + 1.2))
    boxes = axes.boxplot(
        data,
        orientation="horizontal",
        tick_labels=order,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": SURFACE, "linewidth": 2},
        whiskerprops={"color": TEXT_SECONDARY},
        capprops={"color": TEXT_SECONDARY},
    )
    for box in boxes["boxes"]:
        box.set(facecolor=SERIES[0], edgecolor=SERIES[0])
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel("szavak száma (kiugró értékek nélkül)")
    _set_title(axes, title, "a doboz a középső 50%-ot, a fehér vonal a mediánt jelöli")
    return figure


def plot_model_comparison(
    table: pd.DataFrame,
    metric: str = "f1_mean",
    error: str = "f1_std",
    title: str = "Modellek összehasonlítása (5-szörös keresztvalidáció)",
) -> Figure:
    vectorizers = list(dict.fromkeys(table["vektorizáló"]))
    model_order = table.groupby("modell")[metric].max().sort_values().index.tolist()
    figure, axes = _new_figure(7.5, max(3.0, 0.75 * len(model_order) + 1.4))
    height = 0.8 / len(vectorizers)
    for offset, (vectorizer, color) in enumerate(zip(vectorizers, SERIES)):
        subset = table[table["vektorizáló"] == vectorizer].set_index("modell").reindex(model_order)
        positions = [index - 0.4 + height * (offset + 0.5) for index in range(len(model_order))]
        axes.barh(
            positions,
            subset[metric],
            height=height * 0.88,
            xerr=subset[error],
            color=color,
            label=vectorizer,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 2},
        )
    axes.set_yticks(range(len(model_order)), model_order)
    lowest = float((table[metric] - table[error]).min())
    axes.set_xlim(max(0.0, lowest - 0.05), 1.0)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_xlabel("makro F1 (átlag ± szórás)")
    legend = axes.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    _set_title(axes, title, "magasabb = jobb")
    return figure


def plot_share_heatmap(
    table: pd.DataFrame,
    title: str,
    subtitle: str = "cellánként: darabszám és sor szerinti arány",
    xlabel: str = "",
    ylabel: str = "",
    colorbar_label: str = "a sor arányában",
) -> Figure:
    from matplotlib.colors import LinearSegmentedColormap

    counts = table.to_numpy()
    row_share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    colormap = LinearSegmentedColormap.from_list("docproc_blue", ["#f4f8fd", SERIES[0], "#0f3d75"])
    width = 1.0 + 0.85 * table.shape[1]
    height = 1.0 + 0.85 * table.shape[0]
    figure, axes = _new_figure(width + 1.2, height)
    image = axes.imshow(row_share, cmap=colormap, vmin=0, vmax=1)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            color = SURFACE if row_share[row, column] > 0.55 else TEXT_PRIMARY
            label = f"{counts[row, column]}\n{row_share[row, column]:.0%}"
            axes.text(column, row, label, ha="center", va="center", fontsize=8, color=color)
    axes.set_xticks(range(table.shape[1]), list(table.columns), rotation=35, ha="right")
    axes.set_yticks(range(table.shape[0]), list(table.index))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    for spine in axes.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    colorbar.set_label(colorbar_label, color=TEXT_SECONDARY, fontsize=9)
    _set_title(axes, title, subtitle)
    return figure


def plot_confusion_matrix(confusion: pd.DataFrame, title: str = "Tévesztési mátrix (teszthalmaz)") -> Figure:
    return plot_share_heatmap(
        confusion,
        title,
        xlabel="becsült kategória",
        ylabel="valós kategória",
        colorbar_label="a valós kategória arányában",
    )


def plot_top_features(
    features: dict[str, list[tuple[str, float]]],
    title: str = "Legjellemzőbb szavak kategóriánként",
    columns: int = 3,
) -> Figure:
    labels = list(features)
    rows = max(1, -(-len(labels) // columns))
    per_class = max((len(items) for items in features.values()), default=1)
    figure = Figure(figsize=(3.4 * columns, rows * (0.28 * per_class + 1.0) + 0.6), facecolor=SURFACE, layout="constrained")
    grid = figure.subplots(rows, columns, squeeze=False)
    for axes in grid.flat:
        _style_axes(axes)
        axes.set_visible(False)
    for axes, label in zip(grid.flat, labels):
        axes.set_visible(True)
        words = [word for word, _ in features[label]][::-1]
        weights = [weight for _, weight in features[label]][::-1]
        axes.barh(words, weights, height=0.62, color=SERIES[0])
        axes.grid(axis="x", color=GRID, linewidth=0.8)
        axes.set_title(label, loc="left", fontsize=10, fontweight="bold", color=TEXT_PRIMARY)
        axes.tick_params(axis="x", labelsize=8)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    return figure


def plot_k_selection(scores: pd.DataFrame, chosen_k: int | None = None, title: str = "A klaszterszám megválasztása") -> Figure:
    figure = Figure(figsize=(9, 3.6), facecolor=SURFACE, layout="constrained")
    panels = figure.subplots(1, 2)
    for axes, column, label in zip(panels, ("inertia", "silhouette"), ("inercia (könyökmódszer, kisebb = tömörebb)", "sziluett-érték (magasabb = jobb)")):
        _style_axes(axes)
        axes.plot(scores["k"], scores[column], color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        axes.grid(color=GRID, linewidth=0.8)
        axes.set_xlabel("klaszterek száma (k)")
        axes.set_xticks(scores["k"])
        axes.set_title(label, loc="left", fontsize=10, color=TEXT_SECONDARY)
        if chosen_k is not None:
            axes.axvline(chosen_k, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    return figure


def plot_cluster_map(
    coordinates,
    labels,
    titles: dict | None = None,
    title: str = "Dokumentumok két dimenzióban (TruncatedSVD)",
    columns: int = 3,
) -> Figure:
    import numpy as np

    coordinates = np.asarray(coordinates)
    labels = np.asarray(labels)
    groups = sorted(set(labels.tolist()), key=str)
    rows = max(1, -(-len(groups) // columns))
    figure = Figure(figsize=(3.3 * columns, 3.0 * rows + 0.5), facecolor=SURFACE, layout="constrained")
    grid = figure.subplots(rows, columns, squeeze=False, sharex=True, sharey=True)
    for axes in grid.flat:
        _style_axes(axes)
        axes.set_visible(False)
    for axes, group in zip(grid.flat, groups):
        axes.set_visible(True)
        selected = labels == group
        axes.scatter(coordinates[~selected, 0], coordinates[~selected, 1], s=4, color=GRID, linewidths=0)
        axes.scatter(coordinates[selected, 0], coordinates[selected, 1], s=8, color=SERIES[0], alpha=0.7, linewidths=0)
        heading = (titles or {}).get(group, str(group))
        axes.set_title(f"{heading} ({int(selected.sum())} db)", loc="left", fontsize=9, fontweight="bold", color=TEXT_PRIMARY)
        axes.tick_params(labelsize=7)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    return figure
